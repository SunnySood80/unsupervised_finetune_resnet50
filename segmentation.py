import torch
import torch.nn.functional as F

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