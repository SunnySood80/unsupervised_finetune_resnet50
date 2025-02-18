import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import models
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from functools import lru_cache

# Enable faster training configurations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

###############################################################################
#  SmallTransformerEncoder (optional if you need it)
###############################################################################
class SmallTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16, num_layers=2, rank=None):
        super().__init__()
        self.rank = rank if rank is not None else 0
        self.device = torch.device(f'cuda:{self.rank}')
        
        # Increased number of layers and added relative position encoding
        self.num_layers = 4  # Increased from 2
        self.pos_encoding = nn.Parameter(torch.randn(1, 256, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # Doubled feedforward dimension
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Added learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU(inplace=True)

    @torch.amp.autocast('cuda')
    def forward(self, x):
        B, C, H, W = x.shape
        x_reduced = self.pool(x)
        x_flat = x_reduced.flatten(2).transpose(1, 2)
        
        # Add positional encoding
        seq_len = x_flat.shape[1]
        pos_enc = self.pos_encoding[:, :seq_len, :]
        x_flat = x_flat + pos_enc
        
        if self.training:
            def transformer_chunk(x):
                return self.transformer(x / self.temperature.exp())
            x_transformed = checkpoint(transformer_chunk, x_flat, use_reentrant=False)
        else:
            x_transformed = self.transformer(x_flat / self.temperature.exp())
        
        x_restored = x_transformed.transpose(1, 2).view(B, C, H // 2, W // 2)
        x_upsampled = self.upsample(x_restored)
        return x_upsampled

###############################################################################
#  FPNDecoder (optional if you need it)
###############################################################################
class FPNDecoder(nn.Module):
    def __init__(self, in_channels_list=(256, 512, 1024), out_channels=256, rank=None):
        super().__init__()
        self.rank = rank if rank is not None else 0
        
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for in_c in in_channels_list
        ])
        
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            for _ in range(2)
        ])
        
        self.refine_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=8, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(2)
        ])
        
        self.final_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.relu = nn.ReLU(inplace=True)

    @torch.amp.autocast('cuda')
    def forward(self, x1, x2, x3):
        p3 = self.lateral_convs[2](x3)
        p2 = self.lateral_convs[1](x2)
        p1 = self.lateral_convs[0](x1)

        p3_up = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.relu(p2 + p3_up)
        p2 = p2 * self.channel_attention[0](p2)
        p2 = p2 + self.refine_blocks[0](p2)

        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.relu(p1 + p2_up)
        p1 = p1 * self.channel_attention[1](p1)
        p1 = p1 + self.refine_blocks[1](p1)

        p1 = p1 + self.final_enhance(p1)
        return p1

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
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

###############################################################################
#  HybridResNet50FPN
###############################################################################
class HybridResNet50FPN(nn.Module):
    """
    A ResNet50-based feature extractor with a simple FPN top-down pathway.
    """
    def __init__(self, pretrained=True, out_channels=256, rank=None):
        super().__init__()
        self.rank = rank if rank is not None else 0
        
        # Initialize ResNet50 with FCN segmentation head
        self.base_model = models.segmentation.fcn_resnet50(
            weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.base_model = self.base_model.to(memory_format=torch.channels_last)
        
        # Extract backbone layers
        base_resnet = self.base_model.backbone
        self.stem = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool
        )
        self.layer1 = base_resnet.layer1  # 256 channels
        self.layer2 = base_resnet.layer2  # 512 channels
        self.layer3 = base_resnet.layer3  # 1024 channels
        self.layer4 = base_resnet.layer4  # 2048 channels
        
        # Use FCN head instead of custom segmentation head
        self.segmentation_head = self.base_model.classifier
        
        # Better channel adaptation for FCN output
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(21, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_channels, 1)
        )
        
        # Now freeze backbone parameters after layers are initialized
        self._freeze_backbone()
        
        # Initialize optimizer with small learning rate for fine-tuning
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=3e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler for fine-tuning
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,  # Number of iterations
            eta_min=1e-6  # Minimum learning rate
        )
        
        # Loss function for fine-tuning
        self.criterion = DiceLoss(smooth=1.0)  # Better for segmentation tasks
        
        # Rest of initialization...
        self.fpn = FPNDecoder(
            in_channels_list=(256, 512, 1024),
            out_channels=out_channels,
            rank=self.rank
        )
        
        # Add cross-set attention after transformer
        self.cross_attention = CrossSetAttention(
            embed_dim=out_channels,
            num_heads=16
        )
        
        # Add mixing parameter
        self.register_buffer('mixing_alpha', torch.tensor(0.8))  # Start with 80% clustering
        
        # Feature caching
        self.feature_cache = {}
        self.cache_size_limit = 1000

    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        print(f"[HybridResNet50FPN] Freezing backbone layers...")
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False
        print(f"[HybridResNet50FPN] Backbone frozen: stem, layer1-4 frozen")

    def unfreeze_layer(self, layer_name):
        """Unfreeze specific layer parameters"""
        print(f"[HybridResNet50FPN] Unfreezing layer: {layer_name}")
        layer = getattr(self, layer_name)
        for param in layer.parameters():
            param.requires_grad = True
        print(f"[HybridResNet50FPN] Layer {layer_name} unfrozen successfully")
            
    def set_mixing_alpha(self, alpha):
        """Set mixing ratio between clustering and segmentation"""
        print(f"[HybridResNet50FPN] Updating mixing alpha: {self.mixing_alpha.item():.3f} -> {alpha:.3f}")
        self.mixing_alpha = torch.tensor(alpha, device=self.mixing_alpha.device)

    @torch.amp.autocast('cuda')
    def forward(self, x):
        # Regular feature extraction path
        x = x.contiguous(memory_format=torch.channels_last)
        
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Get segmentation and clustering features
        seg_features = self.segmentation_head(x4)
        seg_features = self.channel_adapter(seg_features)
        
        cluster_features = self.fpn(x1, x2, x3)
        cluster_features = self.cross_attention(cluster_features)
        
        # NEW: Compute confidence map from clustering features
        confidence_map = torch.sigmoid(torch.norm(cluster_features, dim=1, keepdim=True))
        mixing_weights = self.mixing_alpha * confidence_map
        
        # Mix features using confidence-aware weighting
        mixed_features = (
            mixing_weights * cluster_features + 
            (1 - mixing_weights) * F.interpolate(
                seg_features, 
                size=cluster_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        )
        
        return mixed_features

    def training_step(self, x, target):
        """Perform a training step for fine-tuning"""
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self(x)
        loss = self.criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        return loss.item()

###############################################################################
#  FilterWeightingSegmenter
###############################################################################
class FilterWeightingSegmenter(nn.Module):
    """
    High-level module that outputs a 256-channel feature map from a ResNet50-FPN.
    """
    def __init__(self, pretrained=True, rank=None, out_channels=256):
        super().__init__()
        self.rank = rank if rank is not None else 0
        
        self.feature_extractor = HybridResNet50FPN(
            pretrained=pretrained, 
            out_channels=out_channels,
            rank=self.rank
        )

    @torch.amp.autocast('cuda')
    def forward(self, x):
        return self.feature_extractor(x)

###############################################################################
#  DDPFeatureExtractor
###############################################################################
class DDPFeatureExtractor(nn.Module):
    """
    Wrapper class for FilterWeightingSegmenter to support DDP training
    """
    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        self.feature_extractor = FilterWeightingSegmenter(
            pretrained=pretrained,
            out_channels=out_channels
        )

    @torch.amp.autocast('cuda')
    def forward(self, x):
        return self.feature_extractor(x)