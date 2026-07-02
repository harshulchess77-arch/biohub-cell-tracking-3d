import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BiohubCellDataset
from model import CellTrackerNet3D

def train_pipeline():
    print("--- Biohub 3D Nueral Network Training Intialize ---")

    #1. Saetup paths and data laoders
    TRAIN_DIR = os.path.join("data", "train")
    zarr_folders = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.zarr')]
    
    if not zarr_folders:
        print("No training data found!")
        return
        
    sample_name = zarr_folders[0]
    print(f"Loading training dataset sample: {sample_name}")
    
    dataset = BiohubCellDataset(sample_name, base_dir=TRAIN_DIR)
    # Using a batch size of 4 to group samples efficiently without blowing up memory
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Instantiate Model, Loss Function, and Optimizer
    model = CellTrackerNet3D()
    criterion = nn.MSELoss() # Standard regression loss: measures squared distance error
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adaptive learning rate optimizer
    
    # Check if a GPU is available to accelerate training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training compute device selected: {device.type.upper()}")
    model.to(device)
    
    # 3. Execute a single training epoch as a pipeline validation check
    print("\nStarting execution check for Epoch 1...")
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        # Move data tensors over to CPU or GPU
        images, targets = images.to(device), targets.to(device)
        
        # Reset the gradients so they don't accumulate across batches
        optimizer.zero_grad()
        
        # Forward pass: Generate location coordinate predictions
        predictions = model(images)
        
        # Compute loss error
        loss = criterion(predictions, targets)
        
        # Backward pass: Compute gradient adjustments
        loss.backward()
        
        # Update model network weights
        optimizer.step()
        
        running_loss += loss.item()
        print(f"   ↳ Batch [{batch_idx+1}/{len(train_loader)}] | Current Batch Loss: {loss.item():.4f}")
        
    print("\n--- TRAINING PIPELINE INITIALIZED SUCCESSFULLY ---")
    print(f"Total Average Checked Loss: {running_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    train_pipeline()