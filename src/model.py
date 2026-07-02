import torch
import torch.nn as nn

class CellTrackerNet3D(nn.Module):
    def __init__(self):
        super(CellTrackerNet3D, self).__init__()
        
        # Convolution Block 1: Extracts low-level edges and textures in 3D
        self.features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # Downsamples shape to (16, 8, 32, 32)
            
            # Convolution Block 2: Extracts complex spatial cell structures
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # Downsamples shape to (32, 4, 16, 16)
        )
        
        # Fully Connected Regression Block: Maps 3D features directly to Z, Y, X coordinates
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3), # Prevents overfitting during competition training
            nn.Linear(128, 3)  # Output: 3 continuous values representing [Z, Y, X]
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

if __name__ == "__main__":
    print("--- 3D Convolutional Neural Network Architecture Test ---")
    
    # Instantiate the network
    model = CellTrackerNet3D()
    print("Model compiled successfully!")
    
    # Simulate a single batch coming out of our dataset loader: Batch size=1, Channels=1, Z=16, Y=64, X=64
    fake_batch = torch.randn(1, 1, 16, 64, 64)
    print(f"Feeding simulated tensor batch through network... Shape: {fake_batch.shape}")
    
    # Run a forward pass
    predictions = model(fake_batch)
    
    print("\n--- MODEL PASS SUCCESSFUL ---")
    print(f"Network Prediction Output Shape: {predictions.shape} (Batch Size, Predicted Coordinates)")
    print(f"Sample Output Raw Coordinates: {predictions.detach().numpy()[0]}")