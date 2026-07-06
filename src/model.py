"""
BioHub Cell Tracking - 3D U-Net Heatmap Segmentation Model

This module implements a lightweight 3D U-Net architecture optimized for
generating 3D probability heatmaps mapping cellular centroids in volumetric
microscopy data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Double convolution block for 3D U-Net."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downsampling with max pooling then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upsampling then double conv."""
    
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size mismatch due to odd dimensions
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CellTrackerUNet3D(nn.Module):
    """
    Lightweight 3D U-Net for cellular centroid heatmap generation.
    
    Architecture:
    - Encoder: 3 downsampling stages with skip connections
    - Bottleneck: Feature extraction at lowest resolution
    - Decoder: 3 upsampling stages with skip connections
    - Output: Single channel probability heatmap
    """
    
    def __init__(self, in_channels=1, out_channels=1, bilinear=False):
        super(CellTrackerUNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv3D(in_channels, 32)
        self.down1 = Down3D(32, 64)
        self.down2 = Down3D(64, 128)
        self.down3 = Down3D(128, 256)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down3D(256, 512 // factor)
        
        # Decoder
        self.up1 = Up3D(512, 256 // factor, bilinear)
        self.up2 = Up3D(256, 128 // factor, bilinear)
        self.up3 = Up3D(128, 64 // factor, bilinear)
        self.up4 = Up3D(64, 32, bilinear)
        
        # Output layer
        self.outc = nn.Conv3d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output heatmap
        logits = self.outc(x)
        return torch.sigmoid(logits)
    
    def predict_centroids(self, heatmap, threshold=0.5, min_distance=5):
        """
        Extract centroid coordinates from predicted heatmap.
        
        Args:
            heatmap: Predicted probability heatmap [B, 1, Z, Y, X]
            threshold: Detection threshold
            min_distance: Minimum distance between peaks
            
        Returns:
            List of centroid coordinates [z, y, x] for each batch
        """
        centroids = []
        
        for b in range(heatmap.shape[0]):
            heatmap_b = heatmap[b, 0].cpu().numpy()
            
            # Simple peak detection (can be improved with more sophisticated methods)
            from scipy.ndimage import maximum_filter
            
            # Local maximum filter
            local_max = maximum_filter(heatmap_b, size=min_distance) == heatmap_b
            
            # Apply threshold
            peak_mask = local_max & (heatmap_b > threshold)
            
            # Get peak coordinates
            peak_coords = np.argwhere(peak_mask)
            
            centroids.append(peak_coords)
        
        return centroids


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in heatmap segmentation.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where p_t is the model's estimated probability for the correct class.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions [B, 1, Z, Y, X], sigmoid activated
            targets: Ground truth heatmaps [B, 1, Z, Y, X]
        """
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation, robust to class imbalance.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions [B, 1, Z, Y, X], sigmoid activated
            targets: Ground truth heatmaps [B, 1, Z, Y, X]
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice loss for robust training with extreme class imbalance.
    """
    
    def __init__(self, focal_weight=0.7, dice_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice_loss = DiceLoss(smooth=1.0)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice


# Legacy alias for backward compatibility
CellTrackerNet3D = CellTrackerUNet3D


if __name__ == "__main__":
    print("--- 3D U-Net Heatmap Segmentation Architecture Test ---")
    
    # Instantiate the network
    model = CellTrackerUNet3D(in_channels=1, out_channels=1)
    print("3D U-Net compiled successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Simulate a batch: Batch size=2, Channels=1, Z=32, Y=128, X=128
    fake_batch = torch.randn(2, 1, 32, 128, 128)
    print(f"\nFeeding simulated tensor batch through network... Shape: {fake_batch.shape}")
    
    # Run a forward pass
    predictions = model(fake_batch)
    
    print("\n--- MODEL PASS SUCCESSFUL ---")
    print(f"Network Prediction Output Shape: {predictions.shape} (Batch, Channels, Z, Y, X)")
    print(f"Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Test loss functions
    print("\n--- TESTING LOSS FUNCTIONS ---")
    targets = torch.rand_like(predictions)
    
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    combined_loss = CombinedLoss()
    
    fl = focal_loss(predictions, targets)
    dl = dice_loss(predictions, targets)
    cl = combined_loss(predictions, targets)
    
    print(f"Focal Loss: {fl.item():.4f}")
    print(f"Dice Loss: {dl.item():.4f}")
    print(f"Combined Loss: {cl.item():.4f}")
    
    # Test centroid extraction
    print("\n--- TESTING CENTROID EXTRACTION ---")
    centroids = model.predict_centroids(predictions, threshold=0.7, min_distance=5)
    print(f"Batch 0: Detected {len(centroids[0])} centroids")
    print(f"Batch 1: Detected {len(centroids[1])} centroids")
    
    print("\n--- 3D U-Net TEST COMPLETE ---")