import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import models
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from functools import lru_cache
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision import transforms

# Enable faster training configurations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add the function definition after the imports but before any classes
def compute_segmentation_map(w_feats: torch.Tensor, binary_mask: torch.Tensor, n_iters: int = 3) -> torch.Tensor:
    """
    Compute binary segmentation map using improved k-means clustering with:
    1. Better centroid initialization
    2. Multiple refinement iterations
    3. Spatial regularization
    4. Confidence thresholding
    
    Args:
        w_feats: Weighted feature tensor [B, C, H, W]
        binary_mask: Binary mask tensor [1, 1, H, W]
        n_iters: Number of k-means iterations
    Returns:
        Binary segmentation map tensor [H, W]
    """
    # Resize binary mask to match feature size
    mask_resized = F.interpolate(
        binary_mask,
        size=(w_feats.shape[2], w_feats.shape[3]),
        mode='nearest'
    )
    
    # Apply binary mask and normalize features
    w_feats = w_feats * mask_resized
    feat_flat = w_feats.squeeze(0).permute(1, 2, 0)  # [H, W, C]
    feat_flat = F.normalize(feat_flat, dim=-1)
    
    # Add spatial coordinates for regularization
    H, W = w_feats.shape[2:]
    y_coords = torch.linspace(-1, 1, H, device=w_feats.device)
    x_coords = torch.linspace(-1, 1, W, device=w_feats.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    spatial_coords = torch.stack([yy, xx], dim=-1) * 0.1  # Scale factor for spatial influence
    
    # Combine features with spatial coordinates
    feat_spatial = torch.cat([
        feat_flat,
        spatial_coords
    ], dim=-1)
    
    # Flatten for clustering
    N = H * W
    D = feat_spatial.shape[-1]
    feat_spatial = feat_spatial.reshape(N, D)
    
    # Initialize centroids using k-means++
    centroids = torch.zeros((2, D), device=w_feats.device)
    
    # First centroid: maximum feature response
    feat_norms = torch.norm(feat_flat.reshape(N, -1), dim=1)
    first_idx = torch.argmax(feat_norms)
    centroids[0] = feat_spatial[first_idx]
    
    # Second centroid: furthest point from first
    dists = torch.norm(feat_spatial - centroids[0], dim=1)
    second_idx = torch.argmax(dists)
    centroids[1] = feat_spatial[second_idx]
    
    # K-means iterations
    clusters = None
    for _ in range(n_iters):
        # Compute distances and assign clusters
        dists = torch.cdist(feat_spatial, centroids)
        new_clusters = dists.argmin(dim=1)
        
        if clusters is not None and torch.all(new_clusters == clusters):
            break
            
        clusters = new_clusters
        
        # Update centroids
        for i in range(2):
            mask = (clusters == i)
            if mask.any():
                centroids[i] = feat_spatial[mask].mean(dim=0)
    
    # Reshape clusters to spatial dimensions
    seg_map = clusters.reshape(H, W).float()
    
    # Compute confidence scores
    dists = torch.cdist(feat_spatial, centroids)
    confidence = torch.abs(dists[:, 0] - dists[:, 1])
    confidence = confidence.reshape(H, W)
    
    # Apply confidence thresholding
    conf_threshold = torch.quantile(confidence[mask_resized.squeeze().bool()], 0.2)
    uncertain_mask = confidence < conf_threshold
    
    # Refine uncertain regions using spatial consistency
    kernel_size = 3
    padding = kernel_size // 2
    for _ in range(2):  # Number of refinement iterations
        # Compute local majority vote
        seg_map_padded = F.pad(seg_map.unsqueeze(0).unsqueeze(0), 
                              (padding, padding, padding, padding), 
                              mode='reflect')
        neighbors = F.unfold(seg_map_padded, kernel_size=kernel_size)
        neighbors = neighbors.reshape(1, kernel_size*kernel_size, H, W)
        local_sum = neighbors.sum(dim=1).squeeze()
        local_vote = (local_sum > (kernel_size*kernel_size/2)).float()
        
        # Update uncertain regions
        seg_map[uncertain_mask] = local_vote[uncertain_mask]
    
    # Final mask application
    seg_map = seg_map * mask_resized.squeeze()
    
    return seg_map

###############################################################################
#  CrossSetAttention
###############################################################################
class CrossSetAttention(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.pool = nn.AdaptiveAvgPool2d((32, 32))
        
        self.reduced_dim = embed_dim // 2
        
        self.q_proj = nn.Linear(embed_dim, self.reduced_dim)
        self.k_proj = nn.Linear(embed_dim, self.reduced_dim)
        self.v_proj = nn.Linear(embed_dim, self.reduced_dim)
        self.out_proj = nn.Linear(self.reduced_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        x_pooled = self.pool(x)
        _, _, H_pooled, W_pooled = x_pooled.shape
        
        x_flat = x_pooled.reshape(B, C, -1).transpose(1, 2)
        x_norm = self.norm1(x_flat)
        
        q = self.q_proj(x_norm).reshape(B, H_pooled*W_pooled, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(B, H_pooled*W_pooled, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(B, H_pooled*W_pooled, self.num_heads, -1).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, H_pooled*W_pooled, self.reduced_dim)
        
        out = self.out_proj(out)
        out = self.norm2(out + x_flat)
        
        out = out.transpose(1, 2).reshape(B, C, H_pooled, W_pooled)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        return out

###############################################################################
#  DiceLoss
###############################################################################
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Get shapes
        B, C, H, W = predictions.shape
        
        # Ensure inputs are the same shape
        if predictions.shape != targets.shape:
            targets = F.interpolate(
                targets.float(),
                size=(H, W),
                mode='nearest'
            )
        
        # Ensure both tensors are contiguous and on same device
        predictions = predictions.contiguous()
        targets = targets.to(predictions.device).contiguous()
        
        # Ensure predictions requires grad
        if not predictions.requires_grad:
            predictions.requires_grad_(True)
        
        # Flatten predictions and targets using reshape
        predictions = predictions.view(B, -1)
        targets = targets.view(B, -1)
        
        # Compute intersection and union per batch
        intersection = (predictions * targets).sum(dim=1)
        union = predictions.sum(dim=1) + targets.sum(dim=1)
        
        # Compute Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return mean loss across batch
        return 1 - dice.mean()

###############################################################################
#  SAM2FeatureExtractor
###############################################################################
class SAM2FeatureExtractor(nn.Module):
    """
    Uses SAM2 (Segment Anything Model 2) as the backbone for feature extraction
    - Extracts both high-level and low-level features
    - Combines them using a learned mixing ratio (self.mixing_alpha)
    - Includes augmentation transforms for consistency training
    """
    def __init__(self, pretrained=True, out_channels=256, rank=None, batch_size=64):
        super().__init__()
        self.rank = rank if rank is not None else 0
        self.device = torch.device(f"cuda:{self.rank}")
        self.batch_size = batch_size
        
        # Initialize SAM2
        config_file = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt" if pretrained else None
        
        # Move model to correct device immediately after creation
        self.model = build_sam2(config_file=config_file, ckpt_path=checkpoint)
        self.model.to(self.device)


        self.image_encoder = self.model.image_encoder
        self.mask_decoder = self.model.sam_mask_decoder  # It's 'sam_mask_decoder' not 'decoder'

        # Get encoder channels - use fixed size since SAM2 uses 256 channels
        encoder_channels = 256
        
        # Move adapter to correct device
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(encoder_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ).to(self.device)
        
        # Add Dice Loss
        self.criterion = DiceLoss(smooth=1.0)
        
        # Learning rate configuration
        self.lr_config = {
            'early_layers': 1e-5,  # Early layers
            'late_layers': 1e-5,   # Late layers
            'adapter': 1e-5        # Feature adapter
        }
        
        # Initially freeze all parameters
        self._freeze_all()
        
        # Add mixing parameter
        self.register_buffer('mixing_alpha', torch.tensor(0.8))
        
        # Initialize optimizer with simple parameter groups
        param_groups = [
            {
                'params': list(self.image_encoder.parameters()),
                'lr': self.lr_config['late_layers']
            },
            {
                'params': self.feature_adapter.parameters(),
                'lr': self.lr_config['adapter']
            }
        ]

        self.optimizer = torch.optim.AdamW(param_groups)

        # # Add augmentation transforms
        # self.augment_transforms = nn.Sequential(
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomAffine(
        #         degrees=15, 
        #         translate=(0.1, 0.1),
        #         scale=(0.9, 1.1)
        #     )
        # )

        # Add binary_mask as a buffer
        self.register_buffer('binary_mask', None)

    def _freeze_all(self):
        """Freeze all parameters initially"""
        print(f"[SAM2FeatureExtractor] Freezing all parameters...")
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.feature_adapter.parameters():
            param.requires_grad = False
        print(f"[SAM2FeatureExtractor] All parameters frozen")

    def unfreeze_layer(self, layer_name):
        """Unfreeze layer and update optimizer"""
        print(f"[SAM2FeatureExtractor] Unfreezing layer: {layer_name}")
        
        # Access the actual trunk (Hiera backbone) of SAM2
        trunk = self.image_encoder.trunk
        
        # Set learning rate based on layer
        if layer_name in ["layer1", "layer2"]:
            current_lr = self.lr_config['early_layers']
        else:  # layer3 or layer4
            current_lr = self.lr_config['late_layers']
        
        # Map layer names to Hiera blocks
        # Note: layer1 (patch_embed, blocks[0,1]) stays permanently frozen
        layer_to_block = {
            'layer1': ['patch_embed', 'blocks.0', 'blocks.1'],  # Stays frozen (earliest visual features)
            'layer2': ['blocks.2', 'blocks.3'],                 # Unfrozen at step 512
            'layer3': ['blocks.4', 'blocks.5'],                 # Unfrozen at step 128
            'layer4': ['blocks.6', 'blocks.7']                  # Unfrozen at step 32
        }
        
        # Unfreeze corresponding blocks
        blocks_to_unfreeze = layer_to_block[layer_name]
        for block_name in blocks_to_unfreeze:
            if block_name == 'patch_embed':
                for param in trunk.patch_embed.parameters():
                    param.requires_grad = True
            else:
                block_idx = int(block_name.split('.')[1])
                for param in trunk.blocks[block_idx].parameters():
                    param.requires_grad = True
            
        # Also unfreeze corresponding FPN (neck) layers
        # Note: neck.convs[0] (corresponding to layer1) stays frozen
        stage_idx = int(layer_name[-1])-1
        if stage_idx < len(self.image_encoder.neck.convs):
            for param in self.image_encoder.neck.convs[stage_idx].parameters():
                param.requires_grad = True
        
        # Reset optimizer with new parameter groups
        self._setup_optimizer()
        
        print(f"[SAM2FeatureExtractor] Layer {layer_name} unfrozen with learning rate: {current_lr:.1e}")

    def set_mixing_alpha(self, alpha):
        """Set the mixing alpha value for feature blending"""
        self.mixing_alpha = torch.tensor(alpha)
        
    def set_binary_mask(self, mask):
        """Set the binary mask buffer"""
        if self.binary_mask is None or (self.binary_mask.shape != mask.shape):
            self.register_buffer('binary_mask', mask)
        else:
            self.binary_mask.copy_(mask)

    @torch.amp.autocast('cuda')
    def forward(self, x, targets=None, augment=False):
        """Forward pass with optional augmentation"""
        if augment and self.training:
            x = self.augment_transforms(x)
            
        # Ensure input is properly formatted and batched
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # Enforce batch size during training
        if self.training:
            B = x.size(0)
            if B > self.batch_size:
                # Process in chunks if input batch is too large
                outputs = []
                for i in range(0, B, self.batch_size):
                    batch = x[i:i + self.batch_size]
                    out = self._forward_chunk(batch, targets[i:i + self.batch_size] if targets is not None else None)
                    outputs.append(out)
                if targets is not None:
                    features, losses = zip(*outputs)
                    return torch.cat(features, dim=0), sum(losses) / len(losses)
                return torch.cat(outputs, dim=0)
            elif B < self.batch_size:
                # Pad batch if too small
                pad_size = self.batch_size - B
                x = torch.cat([x, x[:pad_size]], dim=0)
                if targets is not None:
                    targets = torch.cat([targets, targets[:pad_size]], dim=0)
                out = self._forward_chunk(x, targets)
                if targets is not None:
                    features, loss = out
                    return features[:B], loss
                return out[:B]
        
        return self._forward_chunk(x, targets)

    def _forward_chunk(self, x, targets=None, sample_indices=None):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
            # Get features from image encoder
            features_dict = self.image_encoder(x)
            
            # Get high and low level features
            high_level_features = features_dict['backbone_fpn'][2]
            low_level_features = features_dict['backbone_fpn'][0]
            vision_feats = features_dict['vision_features']
            
            # Add vision features to high level features
            high_level_features = high_level_features + vision_feats
            
            # Process through adapter
            adapted_high = self.feature_adapter(high_level_features)
            
            # Match spatial dimensions
            if low_level_features.shape[-2:] != adapted_high.shape[-2:]:
                adapted_high = F.interpolate(
                    adapted_high,
                    size=low_level_features.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Mix features
            mixed_features = (adapted_high + low_level_features) / 2.0
            
            loss = None
            if targets is not None:
                # Get masks from decoder
                masks, iou_pred = self.mask_decoder(
                    image_embeddings=mixed_features,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=torch.zeros(mixed_features.shape[0], 0, 256).to(self.device),
                    dense_prompt_embeddings=torch.zeros_like(mixed_features),
                    multimask_output=False
                )
                
                self.latest_segmentation = {
                    'masks': masks.squeeze(1),
                    'indices': sample_indices
                }
                
                # CHOOSE instead of mix based on probability
                if torch.rand(1).item() < self.mixing_alpha:  # Ex: 15% chance use PPO
                    chosen_labels = targets  # Use PPO's k-means clustering
                else:  # Ex: 85% chance use SAM2
                    chosen_labels = self.latest_segmentation['masks']  # Use SAM2's decoder output
                
                loss = self.criterion(mixed_features, chosen_labels)
            
            return mixed_features if loss is None else (mixed_features, loss)

    def finetune_step(self, images, pseudo_labels):
        """Fine-tune SAM2 using pseudo-labels as ground truth"""
        # Ensure inputs are on the correct device
        if images.device != self.device:
            images = images.to(self.device)
        if pseudo_labels.device != self.device:
            pseudo_labels = pseudo_labels.to(self.device)
        
        # Ensure proper shape and type
        if pseudo_labels.dim() == 3:
            pseudo_labels = pseudo_labels.unsqueeze(1)
        pseudo_labels = pseudo_labels.float()
        
        # Remove extra dimension from images if present
        if images.dim() == 5:  # [B, 1, C, H, W]
            images = images.squeeze(1)  # -> [B, C, H, W]
        
        self.optimizer.zero_grad()
        
        # Process in smaller chunks to avoid CUDA memory issues
        chunk_size = min(32, images.shape[0])
        total_loss = 0
        num_chunks = 0
        
        for i in range(0, images.shape[0], chunk_size):
            chunk_images = images[i:i + chunk_size]
            chunk_labels = pseudo_labels[i:i + chunk_size]
            
            # Ensure chunk_images has correct shape [B, C, H, W]
            if chunk_images.dim() > 4:
                chunk_images = chunk_images.squeeze(1)
            
            # Enable gradients for feature extraction
            with torch.set_grad_enabled(True):
                # Get features dictionary from image encoder
                features_dict = self.image_encoder(chunk_images)
                
                # Get high and low level features
                high_level_features = features_dict['backbone_fpn'][2]  # [B, 256, 16, 16]
                low_level_features = features_dict['backbone_fpn'][0]   # [B, 256, 64, 64]
                vision_features = features_dict['vision_features']      # [B, 256, 16, 16]
                
                # Add vision features to high level features
                high_level_features = high_level_features + vision_features
                
                # Process through adapter
                adapted_high = self.feature_adapter(high_level_features)
                
                # Match spatial dimensions if needed
                if low_level_features.shape[-2:] != adapted_high.shape[-2:]:
                    adapted_high = F.interpolate(
                        adapted_high,
                        size=low_level_features.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Mix features using current alpha
                mixed_features = (
                    self.mixing_alpha * adapted_high + 
                    (1 - self.mixing_alpha) * low_level_features
                )
                
                # Ensure mixed_features requires grad
                mixed_features.requires_grad_(True)
                
                # Ensure chunk_labels matches mixed_features dimensions
                if chunk_labels.shape[-2:] != mixed_features.shape[-2:]:
                    chunk_labels = F.interpolate(
                        chunk_labels,
                        size=mixed_features.shape[-2:],
                        mode='nearest'
                    )
                
                # Ensure both tensors have same number of channels
                if chunk_labels.shape[1] != mixed_features.shape[1]:
                    chunk_labels = chunk_labels.repeat(1, mixed_features.shape[1], 1, 1)
                
                
                # Compute loss
                loss = self.criterion(mixed_features, chunk_labels)
                
                # Ensure loss requires grad
                if not loss.requires_grad:
                    print("Warning: Loss does not require grad!")
                    print(f"Loss value: {loss.item()}")
                    
                # Backward pass
                loss.backward()
                
                total_loss += loss.item()
                num_chunks += 1
        
        # Average loss across chunks
        avg_loss = total_loss / num_chunks
        
        # Update parameters
        self.optimizer.step()
        
        return avg_loss

    def _setup_optimizer(self):
        """Setup optimizer with current trainable parameters"""
        # Get all trainable parameters
        param_groups = [
            {
                'params': list(self.image_encoder.parameters()),
                'lr': self.lr_config['late_layers']
            },
            {
                'params': self.feature_adapter.parameters(),
                'lr': self.lr_config['adapter']
            }
        ]
        
        self.optimizer = torch.optim.AdamW(param_groups)

###############################################################################
#  FilterWeightingSegmenter
###############################################################################
class FilterWeightingSegmenter(nn.Module):
    """
    High-level module that uses SAM2 for feature extraction.
    """
    def __init__(self, pretrained=True, rank=None, out_channels=256):
        super().__init__()
        self.rank = rank if rank is not None else 0
        self.device = torch.device(f"cuda:{self.rank}")
        
        self.feature_extractor = SAM2FeatureExtractor(
            pretrained=pretrained, 
            out_channels=out_channels,
            rank=self.rank
        ).to(self.device)

    def forward(self, x):
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)
        return self.feature_extractor(x)