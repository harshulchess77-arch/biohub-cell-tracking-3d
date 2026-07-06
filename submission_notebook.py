"""
BioHub Cell Tracking - Kaggle Competition Submission Notebook
Multi-Scale Scale-Space DoG Tracker with Graph Hygiene Sanitization

This script implements a production-ready cell tracking pipeline optimized for the
BioHub Cell Tracking During Development Kaggle competition.

Key Features:
- Multi-Scale Scale-Space Blob Detection (sigma pairs: 1.0-3.0)
- Physical micrometer distance linking (Z=1.625µm, Y=0.40625µm, X=0.40625µm)
- Hungarian algorithm with 7.0µm gating threshold
- Graph hygiene sanitization (multi-parent removal, self-loop removal, non-consecutive edge removal)
- Memory-efficient sequential processing with gc.collect()
"""

import numpy as np
import dask.array as da
import zarr
import gc
import os
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict
from collections import defaultdict

# ============================================================================
# Kaggle Environment Configuration
# ============================================================================
INPUT_DIR = "/kaggle/input/biohub-cell-tracking-during-development/test"
OUTPUT_PATH = "submission.csv"

# Physical voxel dimensions in micrometers (µm)
VOXEL_SIZE_Z = 1.625  # Z-axis: 1.625 µm/voxel
VOXEL_SIZE_Y = 0.40625  # Y-axis: 0.40625 µm/voxel
VOXEL_SIZE_X = 0.40625  # X-axis: 0.40625 µm/voxel

# Maximum physical distance for cell tracking between frames (µm)
MAX_DISTANCE_THRESHOLD = 7.0  # 7.0 µm strict gating

# ============================================================================
# Core Algorithm Functions
# ============================================================================

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


def voxel_to_physical(coords: np.ndarray) -> np.ndarray:
    """
    Convert voxel coordinates to physical coordinates (µm).
    
    Args:
        coords: Array of voxel coordinates [z, y, x]
        
    Returns:
        Physical coordinates in micrometers [z_phys, y_phys, x_phys]
    """
    physical = coords.copy()
    physical[:, 0] *= VOXEL_SIZE_Z  # Z to µm
    physical[:, 1] *= VOXEL_SIZE_Y  # Y to µm
    physical[:, 2] *= VOXEL_SIZE_X  # X to µm
    return physical


def detect_peaks_multiscale_dog(volume: np.ndarray, threshold: float = 0.5, 
                                 min_distance: int = 5) -> np.ndarray:
    """
    Detect peaks using Multi-Scale Scale-Space Blob Detection.
    Computes scale-space maximum over multiple band-pass sigmas simultaneously
    to extract both large dividing blastomeres and tiny dense cell clusters.
    
    Args:
        volume: 3D volume array [Z, Y, X]
        threshold: Detection threshold
        min_distance: Minimum pixel distance between peaks
        
    Returns:
        Array of detected peak coordinates [z, y, x]
    """
    # Scale-space blob detection: iterate through sigma pairs from 1.0 to 3.0
    sigma_pairs = [(1.0, 3.0), (1.5, 4.5), (2.0, 6.0), (2.5, 7.5), (3.0, 9.0)]
    
    scale_space_response = np.zeros_like(volume)
    
    for sigma_small, sigma_large in sigma_pairs:
        # Apply DoG with current sigma pair
        dog = difference_of_gaussians(volume, sigma_small, sigma_large)
        
        # Normalize to [0, 1]
        dog_norm = (dog - dog.min()) / (dog.max() - dog.min() + 1e-8)
        
        # Take scale-space maximum
        scale_space_response = np.maximum(scale_space_response, dog_norm)
    
    # Apply threshold
    threshold_mask = scale_space_response > threshold
    
    # Local maximum detection
    local_max = maximum_filter(scale_space_response, size=min_distance) == scale_space_response
    
    # Combine threshold and local maximum
    peak_mask = threshold_mask & local_max
    
    # Get peak coordinates
    peak_coords = np.argwhere(peak_mask)
    
    return peak_coords


def link_frames_hungarian(frame1_peaks: np.ndarray, frame2_peaks: np.ndarray) -> List[Tuple[int, int]]:
    """
    Link peaks between consecutive frames using Hungarian algorithm.
    Costs calculated using absolute physical micrometer distances.
    
    Args:
        frame1_peaks: Peak coordinates from frame t [N, 3]
        frame2_peaks: Peak coordinates from frame t+1 [M, 3]
        
    Returns:
        List of (index_frame1, index_frame2) tuples representing links
    """
    if len(frame1_peaks) == 0 or len(frame2_peaks) == 0:
        return []
    
    # Convert to physical coordinates for distance calculation
    phys1 = voxel_to_physical(frame1_peaks)
    phys2 = voxel_to_physical(frame2_peaks)
    
    # Compute pairwise distances in physical space (µm)
    distance_matrix = cdist(phys1, phys2, metric='euclidean')
    
    # Apply distance threshold (set large distances to infinity)
    distance_matrix[distance_matrix > MAX_DISTANCE_THRESHOLD] = np.inf
    
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    # Filter out assignments that exceed threshold
    valid_links = []
    for r, c in zip(row_ind, col_ind):
        if distance_matrix[r, c] <= MAX_DISTANCE_THRESHOLD:
            valid_links.append((r, c))
    
    return valid_links


def sanitize_graph(edges: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Implement graph hygiene sanitization to remove structurally impossible graph defects.
    
    Removes:
    - Multi-parent nodes (a cell can only have 1 parent)
    - Self-looping edges
    - Edges connecting non-consecutive timesteps
    - Duplicate edge declarations
    
    Args:
        edges: Array of edge connections [source_id, target_id]
        nodes: Array of node coordinates [t, z, y, x]
        
    Returns:
        Tuple of (sanitized_edges, sanitized_nodes, error_messages)
    """
    error_messages = []
    
    if len(edges) == 0:
        return edges, nodes, error_messages
    
    # Create node time mapping
    node_times = {}
    for idx, node in enumerate(nodes):
        t, z, y, x = node
        node_times[idx] = t
    
    # Filter 1: Remove self-loops
    self_loops = edges[:, 0] == edges[:, 1]
    if np.any(self_loops):
        error_messages.append(f"Removed {np.sum(self_loops)} self-looping edges")
        edges = edges[~self_loops]
    
    # Filter 2: Remove edges connecting non-consecutive timesteps
    valid_edges = []
    non_consecutive_count = 0
    for edge in edges:
        source_id, target_id = edge
        if source_id in node_times and target_id in node_times:
            t_source = node_times[source_id]
            t_target = node_times[target_id]
            # Only allow edges between consecutive timesteps (difference = 1)
            if abs(t_target - t_source) == 1:
                valid_edges.append(edge)
            else:
                non_consecutive_count += 1
    
    if non_consecutive_count > 0:
        error_messages.append(f"Removed {non_consecutive_count} non-consecutive timestep edges")
    edges = np.array(valid_edges) if valid_edges else np.empty((0, 2))
    
    # Filter 3: Remove multi-parent nodes (keep only the closest parent)
    if len(edges) > 0:
        # Count incoming edges per node
        incoming_count = defaultdict(list)
        for idx, edge in enumerate(edges):
            target_id = edge[1]
            incoming_count[target_id].append(idx)
        
        # For nodes with multiple parents, keep only the closest one
        edges_to_remove = set()
        for target_id, edge_indices in incoming_count.items():
            if len(edge_indices) > 1:
                # Find the edge with smallest physical distance
                if target_id in node_times:
                    distances = []
                    for edge_idx in edge_indices:
                        source_id = edges[edge_idx][0]
                        if source_id in node_times:
                            # Calculate physical distance
                            source_node = nodes[source_id]
                            target_node = nodes[target_id]
                            
                            # Physical distance in µm
                            dz = (source_node[1] - target_node[1]) * VOXEL_SIZE_Z
                            dy = (source_node[2] - target_node[2]) * VOXEL_SIZE_Y
                            dx = (source_node[3] - target_node[3]) * VOXEL_SIZE_X
                            dist = np.sqrt(dz**2 + dy**2 + dx**2)
                            distances.append((dist, edge_idx))
                    
                    if distances:
                        # Sort by distance and keep only the closest
                        distances.sort(key=lambda x: x[0])
                        # Remove all except the first (closest)
                        for _, edge_idx in distances[1:]:
                            edges_to_remove.add(edge_idx)
        
        if edges_to_remove:
            error_messages.append(f"Removed {len(edges_to_remove)} multi-parent edges (kept closest parent)")
            keep_mask = np.array([i not in edges_to_remove for i in range(len(edges))])
            edges = edges[keep_mask]
    
    # Filter 4: Remove duplicate edges
    if len(edges) > 0:
        unique_edges = set(tuple(e) for e in edges)
        if len(unique_edges) < len(edges):
            error_messages.append(f"Removed {len(edges) - len(unique_edges)} duplicate edges")
            edges = np.array(list(unique_edges))
    
    return edges, nodes, error_messages


def track_volume(volume: da.Array, threshold: float = 0.5, 
                 min_distance: int = 5) -> Dict[str, np.ndarray]:
    """
    Track cells across all time frames in a volume.
    
    Args:
        volume: 4D volume array [T, Z, Y, X] (dask array for lazy loading)
        threshold: Detection threshold for peak finding
        min_distance: Minimum pixel distance between peaks
        
    Returns:
        Dictionary containing:
            - 'nodes': Array of node coordinates [t, z, y, x]
            - 'edges': Array of edge connections [source_id, target_id]
            - 'sanitization_messages': List of sanitization messages
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
        
        # Detect peaks using multi-scale DoG
        peaks = detect_peaks_multiscale_dog(frame, threshold, min_distance)
        
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
        
        # Memory hygiene: clear frame
        del frame
        gc.collect()
    
    # Link consecutive frames
    for t in range(t_max - 1):
        frame1_data = frame_peaks_with_ids[t]
        frame2_data = frame_peaks_with_ids[t + 1]
        
        if len(frame1_data['peaks']) > 0 and len(frame2_data['peaks']) > 0:
            links = link_frames_hungarian(frame1_data['peaks'], frame2_data['peaks'])
            
            for idx1, idx2 in links:
                source_id = frame1_data['node_ids'][idx1]
                target_id = frame2_data['node_ids'][idx2]
                all_edges.append([source_id, target_id])
            
            print(f"[TRACKER] Frame {t}→{t+1}: Created {len(links)} links")
    
    # Convert to numpy arrays
    nodes_array = np.array(all_nodes)
    edges_array = np.array(all_edges) if all_edges else np.empty((0, 2))
    
    # Apply graph hygiene sanitization
    print(f"[TRACKER] Applying graph hygiene sanitization...")
    edges_sanitized, nodes_sanitized, error_messages = sanitize_graph(edges_array, nodes_array)
    
    for msg in error_messages:
        print(f"[SANITIZER] {msg}")
    
    print(f"[TRACKER] Sanitization complete: {len(edges_array)} → {len(edges_sanitized)} edges")
    
    return {
        'nodes': nodes_sanitized,
        'edges': edges_sanitized,
        'sanitization_messages': error_messages
    }


def process_zarr_dataset(zarr_path: str, threshold: float = 0.5, 
                         min_distance: int = 5) -> Dict[str, np.ndarray]:
    """
    Track cells in a zarr dataset with memory hygiene.
    
    Args:
        zarr_path: Path to zarr dataset
        threshold: Detection threshold for peak finding
        min_distance: Minimum pixel distance between peaks
        
    Returns:
        Dictionary with nodes and edges
    """
    # Open zarr dataset
    zarr_store = zarr.open(zarr_path, mode='r')
    volume = da.from_zarr(zarr_store)
    
    results = track_volume(volume, threshold, min_distance)
    
    # Memory hygiene: close zarr store
    del volume
    del zarr_store
    gc.collect()
    
    return results


def write_submission_csv(results: Dict[str, np.ndarray], dataset_id: str, 
                         output_path: str = OUTPUT_PATH):
    """
    Write tracking results to Kaggle-compliant submission CSV.
    
    Args:
        results: Dictionary containing 'nodes' and 'edges'
        dataset_id: Dataset identifier
        output_path: Path to output CSV file
    """
    nodes = results['nodes']
    edges = results['edges']
    
    sub_records = []
    row_id = 0
    
    # Node rows: coordinates populated, source/target padded with -1
    for idx, node in enumerate(nodes):
        t, z, y, x = node
        sub_records.append({
            "id": row_id,
            "dataset": dataset_id,
            "row_type": "node",
            "node_id": int(idx),
            "t": int(t),
            "z": float(z),
            "y": float(y),
            "x": float(x),
            "source_id": -1,  # Padded for node rows
            "target_id": -1   # Padded for node rows
        })
        row_id += 1
    
    # Edge rows: source/target populated, coordinates padded with -1
    for edge in edges:
        source_id, target_id = edge
        sub_records.append({
            "id": row_id,
            "dataset": dataset_id,
            "row_type": "edge",
            "node_id": -1,      # Padded for edge rows
            "t": -1,           # Padded for edge rows
            "z": -1.0,         # Padded for edge rows
            "y": -1.0,         # Padded for edge rows
            "x": -1.0,         # Padded for edge rows
            "source_id": int(source_id),
            "target_id": int(target_id)
        })
        row_id += 1
    
    # Write to CSV
    import pandas as pd
    sub_df = pd.DataFrame(sub_records)
    
    # Check if file exists to determine write mode
    if os.path.exists(output_path):
        # Append mode (skip header)
        sub_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        # Write mode (include header)
        sub_df.to_csv(output_path, mode='w', header=True, index=False)
    
    print(f"[EXPORT] Wrote {len(sub_df)} rows for dataset {dataset_id}")


# ============================================================================
# Main Execution Pipeline
# ============================================================================

def main():
    """
    Main execution pipeline for Kaggle submission.
    Processes all test datasets sequentially with memory hygiene.
    """
    print("=" * 60)
    print("BioHub Cell Tracking - Kaggle Submission Pipeline")
    print("=" * 60)
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"Physical Dimensions: Z={VOXEL_SIZE_Z}µm, Y={VOXEL_SIZE_Y}µm, X={VOXEL_SIZE_X}µm")
    print(f"Distance Threshold: {MAX_DISTANCE_THRESHOLD}µm")
    print("=" * 60)
    
    # Check input directory
    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        return
    
    # Get list of zarr datasets
    zarr_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.zarr')]
    
    if not zarr_files:
        print(f"[ERROR] No zarr files found in {INPUT_DIR}")
        return
    
    print(f"[INFO] Found {len(zarr_files)} datasets to process")
    print()
    
    # Process each dataset sequentially
    for idx, zarr_file in enumerate(zarr_files):
        dataset_id = zarr_file.replace('.zarr', '')
        zarr_path = os.path.join(INPUT_DIR, zarr_file)
        
        print(f"[PROCESSING] Dataset {idx+1}/{len(zarr_files)}: {dataset_id}")
        print(f"[PROCESSING] Path: {zarr_path}")
        
        try:
            # Run tracking with memory hygiene
            results = process_zarr_dataset(
                zarr_path, 
                threshold=0.5, 
                min_distance=5
            )
            
            # Write to submission CSV
            write_submission_csv(results, dataset_id, OUTPUT_PATH)
            
            # Memory hygiene after each dataset
            del results
            gc.collect()
            
            print(f"[COMPLETE] Dataset {dataset_id} processed successfully")
            print()
            
        except Exception as e:
            print(f"[ERROR] Failed to process {dataset_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("=" * 60)
    print(f"[SUCCESS] All datasets processed. Submission saved to: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Print summary statistics
    if os.path.exists(OUTPUT_PATH):
        import pandas as pd
        sub_df = pd.read_csv(OUTPUT_PATH)
        print(f"[SUMMARY] Total rows in submission: {len(sub_df)}")
        print(f"[SUMMARY] Node rows: {len(sub_df[sub_df['row_type'] == 'node'])}")
        print(f"[SUMMARY] Edge rows: {len(sub_df[sub_df['row_type'] == 'edge'])}")
        print(f"[SUMMARY] Unique datasets: {sub_df['dataset'].nunique()}")


# Execute main pipeline
if __name__ == "__main__":
    main()
