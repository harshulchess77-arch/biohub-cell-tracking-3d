import os
import zarr
import numpy as np
import pandas as pd

def parse_geff_data(dataset_name, base_dir=os.path.join("data", "train")):
    """
    Extracts node coordinates and tracking edges from a split-dimension .geff dataset folder.
    """
    geff_folder = dataset_name.replace(".zarr", ".geff")
    target_path = os.path.join(base_dir, geff_folder)
    
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Could not find .geff tracking folder at: {target_path}")
        
    print(f"Parsing split graph elements from: {target_path}")
    
    # 1. Load the separate coordinate arrays
    t_arr = zarr.open(os.path.join(target_path, "nodes", "props", "t", "values"), mode='r')[:]
    z_arr = zarr.open(os.path.join(target_path, "nodes", "props", "z", "values"), mode='r')[:]
    y_arr = zarr.open(os.path.join(target_path, "nodes", "props", "y", "values"), mode='r')[:]
    x_arr = zarr.open(os.path.join(target_path, "nodes", "props", "x", "values"), mode='r')[:]
    
    # Combine individual 1D coordinate vectors into an (N, 4) spatial matrix
    nodes_matrix = np.column_stack((t_arr, z_arr, y_arr, x_arr))
    
    # 2. Load the edge link matrix (Source ID -> Target ID)
    edges_store = os.path.join(target_path, "edges", "ids")
    edges_matrix = zarr.open(edges_store, mode='r')[:]
    
    return nodes_matrix, edges_matrix

if __name__ == "__main__":
    print("--- Biohub Data Graph Parser ---")
    TRAIN_DIR = os.path.join("data", "train")
    
    try:
        zarr_folders = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.zarr')]
        if zarr_folders:
            sample_name = zarr_folders[0]
            
            nodes, edges = parse_geff_data(sample_name, base_dir=TRAIN_DIR)
            
            print("\n--- GRAPH LOADING SUCCESSFUL ---")
            print(f"Total Detected Cell Nodes: {nodes.shape[0]}")
            print(f"Total Lineage Tracking Edges: {edges.shape[0]}")
            
            if nodes.shape[0] > 0:
                print("\nFirst 3 Ground Truth Cell Centers [Time, Z, Y, X]:")
                print(nodes[:3])
                
            if edges.shape[0] > 0:
                print("\nFirst 3 Lineage Framework Edges [Source_ID, Target_ID]:")
                print(edges[:3])
                
    except Exception as e:
        print(f"Error extracting tracking dimensions: {e}")