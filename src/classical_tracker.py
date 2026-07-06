"""
BioHub Cell Tracking - Anisotropic Classical DoG Tracker

This module implements a fast, zero-weight peak detection framework using
Difference-of-Gaussians (DoG) for cellular centroid detection with physical
microscopy anisotropy correction and Hungarian algorithm temporal linking.
"""

import numpy as np
import dask.array as da
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict
import zarr


def difference_of_gaussians(image, sigma_small, sigma_large):
    """
    Compute Difference-of-Gaussians manually using gaussian_filter.
    
    Args:
        image: Input image
        sigma_small: Small sigma for fine structures
        sigma_large: Large sigma for background
        
    Returns:
        DoG filtered image
    """
    small = gaussian_filter(image, sigma=sigma_small)
    large = gaussian_filter(image, sigma=sigma_large)
    return small - large


class AnisotropicDoGTracker:
    """
    Classical cell tracker using Difference-of-Gaussians peak detection with
    anisotropic scaling for microscopy voxel dimensions and Hungarian algorithm
    for temporal linking.
    """
    
    # Physical voxel dimensions in micrometers (µm)
    VOXEL_SIZE_Z = 1.625  # Z-axis: 1.625 µm/voxel
    VOXEL_SIZE_Y = 0.40625  # Y-axis: 0.40625 µm/voxel
    VOXEL_SIZE_X = 0.40625  # X-axis: 0.40625 µm/voxel
    
    # Maximum physical distance for cell tracking between frames (µm)
    MAX_DISTANCE_THRESHOLD = 8.0  # 8.0 µm
    
    def __init__(self, sigma_small: float = 1.0, sigma_large: float = 3.0, 
                 threshold: float = 0.5, min_distance: int = 5):
        """
        Initialize the DoG tracker.
        
        Args:
            sigma_small: Small sigma for DoG (detects fine structures)
            sigma_large: Large sigma for DoG (detects background)
            threshold: Detection threshold for peak finding
            min_distance: Minimum pixel distance between peaks
        """
        self.sigma_small = sigma_small
        self.sigma_large = sigma_large
        self.threshold = threshold
        self.min_distance = min_distance
        
    def _apply_anisotropic_scaling(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply anisotropic scaling to account for different voxel dimensions.
        Rescales the volume so that physical distances are isotropic.
        
        Args:
            volume: 3D volume array [Z, Y, X]
            
        Returns:
            Anisotropy-corrected volume
        """
        # Calculate scaling factors to make voxels isotropic
        scale_z = self.VOXEL_SIZE_Z / self.VOXEL_SIZE_X
        scale_y = self.VOXEL_SIZE_Y / self.VOXEL_SIZE_X
        scale_x = 1.0  # Reference dimension
        
        # For simplicity, we'll apply scaling during distance calculations
        # rather than resampling the volume (which would be computationally expensive)
        return volume
    
    def _voxel_to_physical(self, coords: np.ndarray) -> np.ndarray:
        """
        Convert voxel coordinates to physical coordinates (µm).
        
        Args:
            coords: Array of voxel coordinates [z, y, x]
            
        Returns:
            Physical coordinates in micrometers [z_phys, y_phys, x_phys]
        """
        physical = coords.copy()
        physical[:, 0] *= self.VOXEL_SIZE_Z  # Z to µm
        physical[:, 1] *= self.VOXEL_SIZE_Y  # Y to µm
        physical[:, 2] *= self.VOXEL_SIZE_X  # X to µm
        return physical
    
    def _physical_to_voxel(self, coords: np.ndarray) -> np.ndarray:
        """
        Convert physical coordinates (µm) to voxel coordinates.
        
        Args:
            coords: Array of physical coordinates [z_phys, y_phys, x_phys]
            
        Returns:
            Voxel coordinates [z, y, x]
        """
        voxel = coords.copy()
        voxel[:, 0] /= self.VOXEL_SIZE_Z  # µm to Z
        voxel[:, 1] /= self.VOXEL_SIZE_Y  # µm to Y
        voxel[:, 2] /= self.VOXEL_SIZE_X  # µm to X
        return voxel
    
    def detect_peaks_dog(self, volume: np.ndarray) -> np.ndarray:
        """
        Detect cellular centroids using Difference-of-Gaussians.
        
        Args:
            volume: 3D volume array [Z, Y, X]
            
        Returns:
            Array of detected peak coordinates [z, y, x]
        """
        # Apply Difference-of-Gaussians
        dog = difference_of_gaussians(volume, self.sigma_small, self.sigma_large)
        
        # Normalize to [0, 1] range
        dog_norm = (dog - dog.min()) / (dog.max() - dog.min() + 1e-8)
        
        # Find peaks above threshold
        threshold_mask = dog_norm > self.threshold
        
        # Simple local maximum detection (can be improved with more sophisticated methods)
        peaks = []
        from scipy.ndimage import maximum_filter
        
        # Local maximum filter
        local_max = maximum_filter(dog_norm, size=self.min_distance) == dog_norm
        
        # Combine threshold and local maximum
        peak_mask = threshold_mask & local_max
        
        # Get peak coordinates
        peak_coords = np.argwhere(peak_mask)
        
        return peak_coords
    
    def detect_peaks_frame(self, volume_frame: np.ndarray) -> np.ndarray:
        """
        Detect peaks in a single time frame.
        
        Args:
            volume_frame: 3D volume for a single time step [Z, Y, X]
            
        Returns:
            Array of detected peak coordinates [z, y, x]
        """
        return self.detect_peaks_dog(volume_frame)
    
    def link_frames_hungarian(self, frame1_peaks: np.ndarray, frame2_peaks: np.ndarray) -> List[Tuple[int, int]]:
        """
        Link peaks between consecutive frames using Hungarian algorithm.
        
        Args:
            frame1_peaks: Peak coordinates from frame t [N, 3]
            frame2_peaks: Peak coordinates from frame t+1 [M, 3]
            
        Returns:
            List of (index_frame1, index_frame2) tuples representing links
        """
        if len(frame1_peaks) == 0 or len(frame2_peaks) == 0:
            return []
        
        # Convert to physical coordinates for distance calculation
        phys1 = self._voxel_to_physical(frame1_peaks)
        phys2 = self._voxel_to_physical(frame2_peaks)
        
        # Compute pairwise distances in physical space (µm)
        distance_matrix = cdist(phys1, phys2, metric='euclidean')
        
        # Apply distance threshold (set large distances to infinity)
        distance_matrix[distance_matrix > self.MAX_DISTANCE_THRESHOLD] = np.inf
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        # Filter out assignments that exceed threshold
        valid_links = []
        for r, c in zip(row_ind, col_ind):
            if distance_matrix[r, c] <= self.MAX_DISTANCE_THRESHOLD:
                valid_links.append((r, c))
        
        return valid_links
    
    def track_volume(self, volume: da.Array) -> Dict[str, np.ndarray]:
        """
        Track cells across all time frames in a volume.
        
        Args:
            volume: 4D volume array [T, Z, Y, X] (dask array for lazy loading)
            
        Returns:
            Dictionary containing:
                - 'nodes': Array of node coordinates [t, z, y, x]
                - 'edges': Array of edge connections [source_id, target_id]
        """
        t_max, z_max, y_max, x_max = volume.shape
        
        all_nodes = []
        all_edges = []
        node_id_counter = 0
        
        # Store peaks for each frame with their node IDs
        frame_peaks_with_ids = []
        
        print(f"[TRACKER] Processing {t_max} time frames...")
        
        for t in range(t_max):
            # Load frame
            frame = volume[t, :, :, :].compute()
            
            # Detect peaks
            peaks = self.detect_peaks_frame(frame)
            
            # Assign node IDs
            frame_node_ids = []
            for peak in peaks:
                z, y, x = peak
                all_nodes.append([t, z, y, x])
                frame_node_ids.append(node_id_counter)
                node_id_counter += 1
            
            frame_peaks_with_ids.append({
                'peaks': peaks,
                'node_ids': frame_node_ids
            })
            
            print(f"[TRACKER] Frame {t}: Detected {len(peaks)} cells")
        
        # Link consecutive frames
        for t in range(t_max - 1):
            frame1_data = frame_peaks_with_ids[t]
            frame2_data = frame_peaks_with_ids[t + 1]
            
            if len(frame1_data['peaks']) > 0 and len(frame2_data['peaks']) > 0:
                links = self.link_frames_hungarian(frame1_data['peaks'], frame2_data['peaks'])
                
                for idx1, idx2 in links:
                    source_id = frame1_data['node_ids'][idx1]
                    target_id = frame2_data['node_ids'][idx2]
                    all_edges.append([source_id, target_id])
                
                print(f"[TRACKER] Frame {t}→{t+1}: Created {len(links)} links")
        
        return {
            'nodes': np.array(all_nodes),
            'edges': np.array(all_edges) if all_edges else np.empty((0, 2))
        }
    
    def track_zarr_dataset(self, zarr_path: str) -> Dict[str, np.ndarray]:
        """
        Track cells in a zarr dataset.
        
        Args:
            zarr_path: Path to zarr dataset
            
        Returns:
            Dictionary with nodes and edges
        """
        # Open zarr dataset
        zarr_store = zarr.open(zarr_path, mode='r')
        volume = da.from_zarr(zarr_store)
        
        return self.track_volume(volume)


def detect_divisions(edges: np.ndarray, nodes: np.ndarray) -> set:
    """
    Detect cell division events from tracking results.
    A division occurs when one parent cell links to two daughter cells.
    
    Args:
        edges: Array of edge connections [source_id, target_id]
        nodes: Array of node coordinates [t, z, y, x]
        
    Returns:
        Set of division events (parent_node_id)
    """
    if len(edges) == 0:
        return set()
    
    # Count outgoing edges per node
    from collections import defaultdict
    outgoing_count = defaultdict(int)
    
    for source_id, target_id in edges:
        outgoing_count[source_id] += 1
    
    # Divisions are nodes with 2+ outgoing edges
    divisions = {node_id for node_id, count in outgoing_count.items() if count >= 2}
    
    return divisions


if __name__ == "__main__":
    print("--- BioHub Classical DoG Tracker Test ---")
    
    # Create synthetic test volume
    test_volume = np.random.rand(10, 32, 128, 128).astype(np.float32)
    
    # Add some synthetic peaks
    test_volume[5, 16, 64, 64] = 10.0
    test_volume[6, 16, 64, 64] = 10.0
    test_volume[5, 20, 80, 80] = 10.0
    
    tracker = AnisotropicDoGTracker(sigma_small=1.0, sigma_large=3.0, threshold=0.7)
    
    # Test peak detection
    peaks = tracker.detect_peaks_frame(test_volume[5])
    print(f"Detected {len(peaks)} peaks in test frame")
    
    # Test linking
    peaks1 = tracker.detect_peaks_frame(test_volume[5])
    peaks2 = tracker.detect_peaks_frame(test_volume[6])
    links = tracker.link_frames_hungarian(peaks1, peaks2)
    print(f"Created {len(links)} links between frames")
    
    # Test anisotropic scaling
    voxel_coords = np.array([[10, 64, 64], [20, 80, 80]])
    physical_coords = tracker._voxel_to_physical(voxel_coords)
    print(f"Voxel to physical conversion:")
    print(f"  Voxel: {voxel_coords}")
    print(f"  Physical (µm): {physical_coords}")
    
    print("\n--- CLASSICAL TRACKER TEST COMPLETE ---")
