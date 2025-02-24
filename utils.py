# ===========================
# Cell 2
# ===========================
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import csv
import os
import copy
from feature_extract import FilterWeightingSegmenter

class MomentumEncoder:
    def __init__(self, model: nn.Module, momentum: float = 0.999, device=None):
        self.momentum = momentum
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create EMA model with same architecture
        if isinstance(model, FilterWeightingSegmenter):
            print("Creating momentum encoder for DDP model...")
            self.ema_model = FilterWeightingSegmenter(
                pretrained=True,
                rank=0 if device is None else device.index,
                out_channels=256  # Match the output channels from SAM2
            )
            
            # Copy binary mask from original model if it exists
            if hasattr(model, 'module'):
                source_model = model.module
            else:
                source_model = model
                
            if hasattr(source_model.feature_extractor, 'binary_mask'):
                self.ema_model.feature_extractor.set_binary_mask(
                    source_model.feature_extractor.binary_mask
                )
                
            # Get state dict excluding binary_mask
            state_dict = source_model.state_dict()
            # Filter out binary_mask from state dict
            filtered_state_dict = {
                k: v for k, v in state_dict.items() 
                if not k.endswith('binary_mask')
            }
            
            # Load filtered state dict
            self.ema_model.load_state_dict(filtered_state_dict, strict=False)
            
        else:
            print("Creating momentum encoder for standard model...")
            # Create new instance with same parameters as original model
            if hasattr(model, 'module'):
                base_model = model.module
            else:
                base_model = model
                
            # Create new instance with same constructor arguments
            self.ema_model = type(base_model)(
                pretrained=True
            )
            
            # Get state dict excluding binary_mask
            state_dict = base_model.state_dict()
            filtered_state_dict = {
                k: v for k, v in state_dict.items() 
                if not k.endswith('binary_mask')
            }
            
            # Load filtered state dict
            self.ema_model.load_state_dict(filtered_state_dict, strict=False)
            
        self.ema_model = self.ema_model.to(self.device)
        self.ema_model.eval()
        
        # Disable gradients for momentum encoder
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update momentum encoder weights"""
        source_model = model.module if hasattr(model, 'module') else model
        for ema_param, param in zip(self.ema_model.parameters(), source_model.parameters()):
            # Ensure both tensors are on the same device
            param_data = param.data.to(self.device)
            ema_param.data = self.momentum * ema_param.data + (1 - self.momentum) * param_data
            
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features - x should already be feature maps"""
        return x.to(self.device)  # Ensure input is on correct device

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def compute_feature_diversity(features: torch.Tensor, batch_size: int = 64) -> float:
    """
    Improve diversity computation to better separate cancer/non-cancer regions:
    - Add structural prior that cancer regions tend to be more heterogeneous
    - Consider local neighborhood statistics
    """
    # Ensure features are batched
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    # Get dimensions
    B, C, H, W = features.shape
    
    # Reshape features to [B, C*H*W]
    features_flat = features.reshape(B, -1)
    
    # Normalize features
    features_flat = F.normalize(features_flat, p=2, dim=1)
    
    total_similarity = 0
    count = 0
    
    # Process in larger batches
    for i in range(0, B, batch_size):
        batch_end = min(i + batch_size, B)
        current_batch = features_flat[i:batch_end]
        
        # Use larger batches for more stable diversity computation
        for j in range(0, B, batch_size):
            j_end = min(j + batch_size, B)
            other_batch = features_flat[j:j_end]
            
            # Compute batch similarity
            batch_similarity = torch.mm(current_batch, other_batch.t())
            
            # Don't count self-similarities
            if i == j:
                batch_similarity.fill_diagonal_(0)
            
            total_similarity += batch_similarity.sum().item()
            count += (batch_end - i) * (j_end - j)
            if i == j:
                count -= (batch_end - i)  # Subtract diagonal count
    
    # Compute average similarity
    avg_similarity = total_similarity / max(count, 1)
    
    # Convert similarity to diversity (higher is better)
    diversity = 1 - avg_similarity
    
    # Add structural prior weighting
    local_variance = compute_local_feature_variance(features)  # Higher for cancer regions
    diversity = diversity * (1 + local_variance)  # Boost diversity score for heterogeneous regions
    
    return diversity

def compute_feature_statistics(features):
    """
    Compute basic statistics of features in a memory-efficient way
    """
    # Move to CPU for statistics
    features = features.cpu()
    
    mean = features.mean().item()
    std = features.std().item()
    
    return {
        'mean': mean,
        'std': std
    }

# Add a memory tracking decorator for debugging if needed
def track_memory(func):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before {func.__name__}: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            print(f"GPU memory after {func.__name__}: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        return result
    return wrapper

def compute_consistency_reward(original_feats: torch.Tensor, aug_feats_list: List[torch.Tensor]) -> float:
    """
    Compares original features with their augmented versions:
    - Normalizes features for better comparison
    - Computes cosine similarity between original and augmented features
    - Returns average consistency across all augmentations
    """
    # Flatten and normalize original features [B, C*H*W]
    orig_flat = F.normalize(original_feats.reshape(original_feats.size(0), -1), dim=1)
    
    aug_similarities = []
    for aug_feats in aug_feats_list:
        # Flatten and normalize augmented features
        aug_flat = F.normalize(aug_feats.reshape(aug_feats.size(0), -1), dim=1)
        # Compute cosine similarity between original and augmented
        similarity = F.cosine_similarity(orig_flat, aug_flat, dim=1).mean()
        aug_similarities.append(similarity)
    
    return torch.stack(aug_similarities).mean().item()

def compute_boundary_strength(features: torch.Tensor) -> float:
    """
    Compute boundary strength based on feature gradients.
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
        
    B, C, H, W = features.shape
    
    grad_x = torch.zeros_like(features)
    grad_y = torch.zeros_like(features)
    
    grad_x[..., :-1] = features[..., 1:] - features[..., :-1]
    grad_y[..., :-1, :] = features[..., 1:, :] - features[..., :-1, :]
    
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = gradient_magnitude.contiguous().view(B, -1)
    
    if gradient_magnitude.shape[1] > 0:
        gradient_magnitude = F.normalize(gradient_magnitude, dim=1)
        boundary_score = gradient_magnitude.mean().item()
    else:
        boundary_score = 0.0
    
    return float(boundary_score)

def compute_local_coherence(features: torch.Tensor, kernel_size: int = 3) -> float:
    """
    Compute local feature coherence using average pooling.
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    padding = kernel_size // 2
    local_mean = F.avg_pool2d(features, kernel_size=kernel_size, 
                             stride=1, padding=padding)
    
    coherence = -F.mse_loss(features, local_mean)
    return float(coherence.item())

def visualize_map_with_augs(
    image_tensors,
    heatmaps,
    ground_truth,
    binary_mask,
    reward,
    save_path,
    title=None
):
    """
    Visualize segmentation maps with augmentations
    """
    # Prepare binary mask
    mask_resized = F.interpolate(
        binary_mask,
        size=(256, 256),
        mode='nearest'
    ).squeeze().cpu().numpy()
    
    n_images = len(image_tensors)
    n_cols = 4
    n_rows = n_images
    
    plt.figure(figsize=(20, 5*n_rows))
    
    for idx, (img, hmap) in enumerate(zip(image_tensors, heatmaps)):
        if img.dim() == 4:
            img = img.squeeze(0)
            
        # Resize image
        img_resized = F.interpolate(
            img.unsqueeze(0), 
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Resize heatmap and apply mask
        hmap_resized = F.interpolate(
            hmap.unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode='nearest'
        ).squeeze().detach().cpu().numpy()
        hmap_resized = hmap_resized * mask_resized  # Apply binary mask
        
        img_np = img_resized.detach().cpu().permute(1,2,0).numpy()
        
        # Original Image
        plt.subplot(n_rows, n_cols, idx*n_cols + 1)
        plt.imshow(img_np)
        plt.title(f"{'Original' if idx==0 else f'Aug {idx}'}")
        plt.axis("off")
        
        # Ground Truth
        plt.subplot(n_rows, n_cols, idx*n_cols + 2)
        plt.imshow(ground_truth, cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")
        
        # Binary Map (black/white)
        plt.subplot(n_rows, n_cols, idx*n_cols + 3)
        plt.imshow(hmap_resized, cmap='gray')
        plt.title(f"Binary Map (R={reward:.3f})")
        plt.axis("off")
        
        # Overlay - using resized versions of both image and heatmap
        plt.subplot(n_rows, n_cols, idx*n_cols + 4)
        plt.imshow(img_np)
        plt.imshow(hmap_resized, cmap='gray', alpha=0.6)
        plt.title("Overlay")
        plt.axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def compute_local_feature_variance(features: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Compute local variance of features in a neighborhood.
    Higher variance indicates more heterogeneous regions (likely cancer).
    
    Args:
        features: Tensor of shape [B, C, H, W]
        kernel_size: Size of local neighborhood
    Returns:
        Local variance map
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
        
    B, C, H, W = features.shape
    
    # Compute local mean
    padding = kernel_size // 2
    local_mean = F.avg_pool2d(
        features,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )
    
    # Compute local variance
    local_var = F.avg_pool2d(
        (features - local_mean)**2,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )
    
    # Normalize variance
    local_var = local_var.mean(dim=1, keepdim=True)  # Average across channels
    local_var = (local_var - local_var.min()) / (local_var.max() - local_var.min() + 1e-8)
    
    return local_var.squeeze(1).mean()  # Return scalar variance score