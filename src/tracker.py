import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

def link_time_frames(nodes_at_t0, nodes_at_t1, max_distance=30):
    """
    Connects cells from one frame to the next frame based on closest 
    spatial proximity. Returns an array of predicted tracking edges.
    """
    if len(nodes_at_t0) == 0 or len(nodes_at_t1) == 0:
        return np.empty((0, 2))
        
    # Extract only the spatial coordinates (Z, Y, X) for distance matching
    coords_t0 = nodes_at_t0[:, 1:4]
    coords_t1 = nodes_at_t1[:, 1:4]
    
    # Calculate a matrix of distances between all cells in t0 and all cells in t1
    dist_mat = distance_matrix(coords_t0, coords_t1)
    
    predicted_edges = []
    
    # Find the closest pairs
    for i in range(len(coords_t0)):
        closest_idx = np.argmin(dist_mat[i])
        closest_dist = dist_mat[i, closest_idx]
        
        # Only create a tracking link if the cell didn't move an impossible distance
        if closest_dist <= max_distance:
            # Pair the original row index or unique ID
            predicted_edges.append([i, closest_idx])
            
    return np.array(predicted_edges)

if __name__ == "__main__":
    print("--- Biohub Frame-to-Frame Lineage Tracker Test ---")
    from graph_loader import parse_geff_data
    import os
    
    TRAIN_DIR = os.path.join("data", "train")
    zarr_folders = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.zarr')]
    
    if zarr_folders:
        sample_name = zarr_folders[0]
        nodes, _ = parse_geff_data(sample_name, base_dir=TRAIN_DIR)
        
        # Separate out cell coordinates belonging strictly to Frame 0 vs Frame 1
        nodes_t0 = nodes[nodes[:, 0] == 0]
        nodes_t1 = nodes[nodes[:, 0] == 1]
        
        print(f"\nCells detected in Frame 0: {len(nodes_t0)}")
        print(f"Cells detected in Frame 1: {len(nodes_t1)}")
        
        # Run our linkage engine
        discovered_links = link_time_frames(nodes_t0, nodes_t1)
        
        print("\n--- TRACKING LINKAGE SUCCESSFUL ---")
        print(f"Successfully reconstructed {len(discovered_links)} temporal tracking links!")
        if len(discovered_links) > 0:
            print(f"Sample generated tracking link (Cell index in T0 -> Cell index in T1):\n {discovered_links[:2]}")