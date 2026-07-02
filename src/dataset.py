import os
import torch
from torch.utils.data import Dataset
from data_loader import get_zarr_stream
from graph_loader import parse_geff_data
from image_processor import extract_3d_crop

class BiohubCellDataset(Dataset):
    def __init__(self, dataset_name, base_dir=os.path.join("data", "train"), crop_size=(16, 64, 64)):
        """
        Custom PyTorch Dataset that pairs 3D image patches with their respective ground truth coordinates.
        """
        self.crop_size = crop_size
        
        # Connect to the streaming data paths
        self.volume = get_zarr_stream(dataset_name, base_dir=base_dir)
        self.nodes, self.edges = parse_geff_data(dataset_name, base_dir=base_dir)
        
    def __len__(self):
        # The number of unique cell locations available to learn from
        return len(self.nodes)
        
    def __getitem__(self, idx):
        # Extract individual coordinates for the selected index
        t, z, y, x = self.nodes[idx]
        
        # 1. Grab the clean, localized 3D video sub-cube patch
        patch = extract_3d_crop(self.volume, t, z, y, x, crop_size=self.crop_size)
        
        # 2. Convert raw numpy array into a PyTorch Tensor
        # Add a channel dimension (C, Z, Y, X) which deep learning layers expect
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
        
        # 3. Create our target tracking labels (the raw coordinates)
        coords_tensor = torch.tensor([z, y, x], dtype=torch.float32)
        
        return patch_tensor, coords_tensor

if __name__ == "__main__":
    print("--- PyTorch Dataset Pipeline Test ---")
    TRAIN_DIR = os.path.join("data", "train")
    zarr_folders = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.zarr')]
    
    if zarr_folders:
        try:
            cell_dataset = BiohubCellDataset(zarr_folders[0], base_dir=TRAIN_DIR)
            print(f"Dataset total structural samples loaded: {len(cell_dataset)}")
            
            # Pull the first complete learning sample pair out of the conveyor belt pipeline
            img_tensor, coord_target = cell_dataset[0]
            
            print("\n--- PYTORCH PIPELINE SUCCESSFUL ---")
            print(f"Image Tensor Shape: {img_tensor.shape} (Channels, Depth, Height, Width)")
            print(f"Label Vector: {coord_target} (True spatial Z, Y, X location)")
            
        except Exception as e:
            print(f"Error during pipeline execution: {e}")