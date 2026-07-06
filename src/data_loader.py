"""
BioHub Cell Tracking - PyTorch Data Loader with 3D Augmentations

This module provides memory-mapped Zarr data loading and a PyTorch Dataset
class with robust 3D augmentations for training the 3D U-Net segmentation model.
"""

import os
import zarr
import dask.array as da
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import random


def get_zarr_stream(dataset_name, base_dir=os.path.join("data", "train")):
    """
    Opens a 4D Biohub cell tracking dataset without loading the whole file into RAM.
    Returns a dask array mapped to dimensions (Time, Z, Y, X).
    
    Parameters:
    -----------
    dataset_name : str
        The name of the specific .zarr folder (e.g., 'sample_001.zarr')
    base_dir : str
        The parent folder directory where the dataset sits. Default is 'data/train'.
    """
    target_path = os.path.join(base_dir, dataset_name, "0")
    
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Could not find data matrix at: {target_path}")
        
    zarr_store = zarr.open(target_path, mode='r')
    lazy_volume = da.from_zarr(zarr_store)
    return lazy_volume


class CellTrackingDataset(Dataset):
    """
    PyTorch Dataset for 3D cell tracking with memory-mapped Zarr loading and 3D augmentations.
    
    Features:
    - Lazy loading from Zarr files to minimize memory usage
    - 3D spatial augmentations (rotations, flips)
    - Intensity augmentations (scaling, contrast)
    - Ground truth heatmap generation from centroid coordinates
    """
    
    def __init__(self, 
                 dataset_names: List[str],
                 base_dir: str = "data/train",
                 graph_loader=None,
                 crop_size: Tuple[int, int, int] = (32, 128, 128),
                 augment: bool = True,
                 heatmap_sigma: float = 2.0):
        """
        Args:
            dataset_names: List of zarr dataset names
            base_dir: Base directory containing datasets
            graph_loader: Function to load ground truth graphs
            crop_size: Size of 3D crops (Z, Y, X)
            augment: Whether to apply data augmentations
            heatmap_sigma: Sigma for Gaussian heatmap generation
        """
        self.dataset_names = dataset_names
        self.base_dir = base_dir
        self.graph_loader = graph_loader
        self.crop_size = crop_size
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma
        
        # Load all volumes and ground truth
        self.volumes = []
        self.nodes_list = []
        
        print(f"[DATASET] Loading {len(dataset_names)} datasets...")
        for name in dataset_names:
            volume = get_zarr_stream(name, base_dir)
            self.volumes.append(volume)
            
            if graph_loader:
                try:
                    nodes, edges = graph_loader(name, base_dir=base_dir)
                    self.nodes_list.append(nodes)
                except:
                    self.nodes_list.append(None)
            else:
                self.nodes_list.append(None)
        
        # Build sample index (frame, dataset_idx)
        self.samples = []
        for d_idx, volume in enumerate(self.volumes):
            t_max = volume.shape[0]
            for t in range(t_max):
                self.samples.append((t, d_idx))
        
        print(f"[DATASET] Total samples: {len(self.samples)}")
    
    def _generate_heatmap(self, shape: Tuple[int, int, int], 
                         centroids: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian heatmap from centroid coordinates.
        
        Args:
            shape: Heatmap shape (Z, Y, X)
            centroids: Array of centroid coordinates [z, y, x]
            
        Returns:
            Gaussian heatmap
        """
        heatmap = np.zeros(shape, dtype=np.float32)
        
        if len(centroids) == 0:
            return heatmap
        
        z_grid, y_grid, x_grid = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        
        for z, y, x in centroids:
            # Ensure coordinates are within bounds
            z = np.clip(z, 0, shape[0] - 1)
            y = np.clip(y, 0, shape[1] - 1)
            x = np.clip(x, 0, shape[2] - 1)
            
            # Gaussian blob
            dist_sq = ((z_grid - z)**2 + (y_grid - y)**2 + (x_grid - x)**2)
            blob = np.exp(-dist_sq / (2 * self.heatmap_sigma**2))
            heatmap = np.maximum(heatmap, blob)
        
        return heatmap
    
    def _random_rotation_90(self, volume: np.ndarray) -> np.ndarray:
        """Apply random 90-degree rotation in Y-X plane."""
        k = random.randint(0, 3)
        if k > 0:
            volume = np.rot90(volume, k=k, axes=(1, 2))
        return volume
    
    def _random_flip_z(self, volume: np.ndarray) -> np.ndarray:
        """Random flip along Z-axis (depth)."""
        if random.random() > 0.5:
            volume = volume[::-1, :, :]
        return volume
    
    def _random_intensity_scale(self, volume: np.ndarray) -> np.ndarray:
        """Random intensity scaling."""
        scale = random.uniform(0.8, 1.2)
        volume = volume * scale
        return volume
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range."""
        volume = volume.astype(np.float32)
        p2, p98 = np.percentile(volume, (2, 98))
        volume = np.clip(volume, p2, p98)
        volume = (volume - p2) / (p98 - p2 + 1e-8)
        return volume
    
    def _extract_crop(self, volume: da.Array, t: int, 
                     center_z: int, center_y: int, center_x: int) -> np.ndarray:
        """
        Extract 3D crop around a center point.
        
        Args:
            volume: Dask volume array
            t: Time frame
            center_z, center_y, center_x: Center coordinates
            
        Returns:
            3D crop as numpy array
        """
        cz, cy, cx = self.crop_size
        z_dim, y_dim, x_dim = volume.shape[1], volume.shape[2], volume.shape[3]
        
        z_start = max(0, min(center_z - cz // 2, z_dim - cz))
        y_start = max(0, min(center_y - cy // 2, y_dim - cy))
        x_start = max(0, min(center_x - cx // 2, x_dim - cx))
        
        crop = volume[t, z_start:z_start+cz, y_start:y_start+cy, x_start:x_start+cx].compute()
        return crop
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample with augmentations.
        
        Returns:
            (volume, heatmap) tuple as torch tensors
        """
        t, d_idx = self.samples[idx]
        volume = self.volumes[d_idx]
        nodes = self.nodes_list[d_idx]
        
        z_dim, y_dim, x_dim = volume.shape[1], volume.shape[2], volume.shape[3]
        cz, cy, cx = self.crop_size
        
        # Extract random crop
        if nodes is not None and len(nodes) > 0:
            # Extract crop around a random cell centroid
            frame_nodes = nodes[nodes[:, 0] == t]
            if len(frame_nodes) > 0:
                random_node = frame_nodes[random.randint(0, len(frame_nodes) - 1)]
                _, nz, ny, nx = random_node
                crop = self._extract_crop(volume, t, int(nz), int(ny), int(nx))
            else:
                # Random crop if no cells in frame
                z_start = random.randint(0, z_dim - cz)
                y_start = random.randint(0, y_dim - cy)
                x_start = random.randint(0, x_dim - cx)
                crop = volume[t, z_start:z_start+cz, y_start:y_start+cy, x_start:x_start+cx].compute()
        else:
            # Random crop if no ground truth
            z_start = random.randint(0, z_dim - cz)
            y_start = random.randint(0, y_dim - cy)
            x_start = random.randint(0, x_dim - cx)
            crop = volume[t, z_start:z_start+cz, y_start:y_start+cy, x_start:x_start+cx].compute()
        
        # Normalize
        crop = self._normalize_volume(crop)
        
        # Generate heatmap
        if nodes is not None and len(nodes) > 0:
            frame_nodes = nodes[nodes[:, 0] == t]
            # Adjust node coordinates to crop coordinates
            z_start = max(0, min(z_dim // 2 - cz // 2, z_dim - cz))
            y_start = max(0, min(y_dim // 2 - cy // 2, y_dim - cy))
            x_start = max(0, min(x_dim // 2 - cx // 2, x_dim - cx))
            
            centroids = []
            for node in frame_nodes:
                _, nz, ny, nx = node
                nz_adj = nz - z_start
                ny_adj = ny - y_start
                nx_adj = nx - x_start
                if 0 <= nz_adj < cz and 0 <= ny_adj < cy and 0 <= nx_adj < cx:
                    centroids.append([nz_adj, ny_adj, nx_adj])
            
            heatmap = self._generate_heatmap(self.crop_size, np.array(centroids))
        else:
            heatmap = np.zeros(self.crop_size, dtype=np.float32)
        
        # Apply augmentations
        if self.augment:
            crop = self._random_rotation_90(crop)
            heatmap = self._random_rotation_90(heatmap)
            
            crop = self._random_flip_z(crop)
            heatmap = self._random_flip_z(heatmap)
            
            crop = self._random_intensity_scale(crop)
        
        # Convert to tensors
        crop_tensor = torch.from_numpy(crop).unsqueeze(0).float()  # Add channel dim
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).float()  # Add channel dim
        
        return crop_tensor, heatmap_tensor


def create_dataloaders(train_datasets: List[str],
                      val_datasets: List[str],
                      base_dir: str = "data/train",
                      graph_loader=None,
                      batch_size: int = 4,
                      num_workers: int = 2,
                      crop_size: Tuple[int, int, int] = (32, 128, 128)) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_datasets: List of training dataset names
        val_datasets: List of validation dataset names
        base_dir: Base directory containing datasets
        graph_loader: Function to load ground truth graphs
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        crop_size: Size of 3D crops
        
    Returns:
        (train_loader, val_loader) tuple
    """
    train_dataset = CellTrackingDataset(
        train_datasets, base_dir, graph_loader, crop_size, augment=True
    )
    val_dataset = CellTrackingDataset(
        val_datasets, base_dir, graph_loader, crop_size, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("--- BioHub PyTorch Data Loader Test ---")
    
    # Test with synthetic data if no real data available
    print("[TEST] Creating synthetic dataset...")
    
    # Create a simple test
    dummy_volume = np.random.rand(10, 32, 128, 128).astype(np.float32)
    dummy_nodes = np.array([[0, 16, 64, 64], [1, 16, 64, 64]])
    
    print(f"[TEST] Volume shape: {dummy_volume.shape}")
    print(f"[TEST] Nodes shape: {dummy_nodes.shape}")
    
    # Test heatmap generation
    from src.data_loader import CellTrackingDataset
    
    dataset = CellTrackingDataset(
        dataset_names=[],
        base_dir="data/train",
        graph_loader=None,
        crop_size=(32, 128, 128),
        augment=False
    )
    
    # Test heatmap generation
    centroids = np.array([[16, 64, 64], [20, 80, 80]])
    heatmap = dataset._generate_heatmap((32, 128, 128), centroids)
    
    print(f"[TEST] Heatmap shape: {heatmap.shape}")
    print(f"[TEST] Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print(f"[TEST] Heatmap sum: {heatmap.sum():.4f}")
    
    print("\n--- DATA LOADER TEST COMPLETE ---")