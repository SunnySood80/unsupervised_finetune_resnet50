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
    def __init__(self, embed_dim=2048, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
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
        
        # Lateral connections with batch norm
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for in_c in in_channels_list
        ])
        
        # Channel attention modules
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
        
        # Refinement blocks
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

    def forward(self, x1, x2, x3):
        # Process features from bottom up
        p3 = self.lateral_convs[2](x3)
        p2 = self.lateral_convs[1](x2)
        p1 = self.lateral_convs[0](x1)

        # Top-down pathway with attention and refinement
        p3_up = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.relu(p2 + p3_up)
        p2 = p2 * self.channel_attention[0](p2)
        p2 = p2 + self.refine_blocks[0](p2)

        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.relu(p1 + p2_up)
        p1 = p1 * self.channel_attention[1](p1)
        p1 = p1 + self.refine_blocks[1](p1)

        # Final enhancement
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
#  HybridResNet50FPN
###############################################################################
class HybridResNet50FPN(nn.Module):
    """
    A ResNet50-based feature extractor with a simple FPN top-down pathway.
    """
    def __init__(self, pretrained=True, out_channels=256, rank=None):
        super().__init__()
        self.rank = rank if rank is not None else 0
        
        # Initialize ResNet50 backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base_resnet = models.resnet50(weights=weights)
        
        # Start with everything frozen
        for param in base_resnet.parameters():
            param.requires_grad = False
            
        # Extract layers
        self.stem = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool
        )
        self.layer1 = base_resnet.layer1  # 256 channels
        self.layer2 = base_resnet.layer2  # 512 channels
        self.layer3 = base_resnet.layer3  # 1024 channels
        
        # Store layers in order for gradual unfreezing
        self.backbone_layers = nn.ModuleList([self.layer3, self.layer2, self.layer1, self.stem])
        self.current_unfrozen = -1  # No layers unfrozen initially
        
        # Add transformer and FPN
        self.transformer = SmallTransformerEncoder(embed_dim=2048, num_heads=16)
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
        
        # Feature caching
        self.feature_cache = {}
        self.cache_size_limit = 1000

    def unfreeze_next_layer(self):
        """Unfreeze the next layer from top to bottom"""
        if self.current_unfrozen < len(self.backbone_layers) - 1:
            self.current_unfrozen += 1
            layer = self.backbone_layers[self.current_unfrozen]
            for param in layer.parameters():
                param.requires_grad = True
            return True
        return False

    @torch.amp.autocast('cuda')
    def forward(self, x):
        # Check cache
        cache_key = (x.data_ptr(), x.shape)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        x = x.contiguous(memory_format=torch.channels_last)
        
        # Extract features
        x = self.stem(x)
        x1 = self.layer1(x)    # 256 channels
        x2 = self.layer2(x1)   # 512 channels
        x3 = self.layer3(x2)   # 1024 channels

        # Apply transformer to high-level features
        x3 = self.transformer(x3)
        
        # FPN decoder
        p1 = self.fpn(x1, x2, x3)
        
        # Apply cross-set attention
        p1 = self.cross_attention(p1)

        # Cache result
        if len(self.feature_cache) >= self.cache_size_limit:
            self.feature_cache.clear()
        self.feature_cache[cache_key] = p1

        return p1

###############################################################################
#  FilterWeightingSegmenter
###############################################################################
class FilterWeightingSegmenter(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load ResNet50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract layers
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        
        # Store layers for gradual unfreezing
        self.backbone_layers = nn.ModuleList([
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3
        ])
        
        # Add transformer enhancement
        self.transformer = SmallTransformerEncoder(embed_dim=1024, num_heads=16)
        
        # Add FPN decoder
        self.fpn = FPNDecoder(
            in_channels_list=(256, 512, 1024),
            out_channels=256
        )
        
        # Cross attention for global context
        self.cross_attention = CrossSetAttention(
            embed_dim=256,
            num_heads=8
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
        
        self.training_mode = 'feature_extract'
        self.freeze_backbone()  # Start with everything frozen

    def freeze_backbone(self):
        """Freeze the ResNet50 backbone"""
        for layer in self.backbone_layers:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze the ResNet50 backbone for fine-tuning"""
        for layer in self.backbone_layers:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        # Extract features through backbone
        x = self.stem(x)
        f1 = self.layer1(x)    # 256 channels
        f2 = self.layer2(f1)   # 512 channels
        f3 = self.layer3(f2)   # 1024 channels
        
        # Apply transformer to high-level features
        f3 = self.transformer(f3)
        
        # FPN decoder
        features = self.fpn(f1, f2, f3)
        
        # Apply cross attention
        features = self.cross_attention(features)
        
        if self.training_mode == 'feature_extract':
            return features
        else:
            seg_logits = self.seg_head(features)
            return features, seg_logits
    
    def set_training_mode(self, mode):
        """Set whether to do feature extraction or fine-tuning"""
        assert mode in ['feature_extract', 'finetune']
        self.training_mode = mode
        
        if mode == 'feature_extract':
            self.eval()  # Set to eval mode for feature extraction
            # Freeze all parameters
            self.freeze_backbone()
            for param in self.seg_head.parameters():
                param.requires_grad = False
        else:
            self.train()  # Set to train mode for fine-tuning
            # Unfreeze segmentation head only (backbone stays frozen initially)
            for param in self.seg_head.parameters():
                param.requires_grad = True
    
    def unfreeze_next_layer(self):
        """Unfreeze the next frozen layer in the backbone, returns True if a layer was unfrozen"""
        for layer in self.backbone_layers:
            # Check if layer is frozen
            if not any(p.requires_grad for p in layer.parameters()):
                # Unfreeze this layer
                for param in layer.parameters():
                    param.requires_grad = True
                return True
        return False  # No more layers to unfreeze

###############################################################################
#  DDPFeatureExtractor
###############################################################################
class DDPFeatureExtractor(nn.Module):
    """
    Wrapper class for FilterWeightingSegmenter to support DDP training
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.feature_extractor = FilterWeightingSegmenter(pretrained=pretrained)

    @torch.amp.autocast('cuda')
    def forward(self, x):
        return self.feature_extractor(x)
        
    def set_training_mode(self, mode):
        self.feature_extractor.set_training_mode(mode)