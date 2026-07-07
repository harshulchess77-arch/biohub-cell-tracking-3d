"""
BioHub Cell Tracking - Elite Competition Algorithm
Adaptive Multi-Scale Blob Extraction with Velocity-Based Cognitive Tracking

This script implements a state-of-the-art cell tracking pipeline optimized for the
BioHub Cell Tracking During Development Kaggle competition.
"""

import os
import json
import gc
import zlib
import gzip
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict
from scipy.ndimage import gaussian_filter, maximum_filter, distance_transform_edt, label
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# High-performance 3D watershed segmenter from scikit-image (Avoids Scipy ImportError)
from skimage.segmentation import watershed

# Initialize blosc2 safely for Zarr block decompression
try:
    import blosc2
except ImportError:
    pass

# ============================================================================
# Competition Geometry & Tracking Configuration
# ============================================================================
SCALE = np.array([1.625, 0.40625, 0.40625])  # Physical resolution: [Z, Y, X] µm/voxel
MIN_PHYSICAL_SIGMA = 1.2
MAX_PHYSICAL_SIGMA = 3.8
NUM_SIGMA_SCALES = 4

# Gating hyper-parameters (in physical micrometers)
MAX_LINK_DISTANCE = 7.0       # Strict edge Jaccard maximum matching gate
GAP_CLOSING_RADIUS = 5.5      # Maximum distance allowed for multi-frame gap bridging
MITOSIS_EXPANSION_RADIUS = 8.0 # Structural bounds for mother-daughter splitting

OUTPUT_PATH = "submission.csv"

# Multi-path search robustness for Kaggle evaluation orchestration engines
CANDIDATE_PATHS = [
    '/kaggle/input/competitions/biohub-cell-tracking-during-development/test',
    '/kaggle/input/biohub-cell-tracking-during-development/test'
]
TEST_DIR = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)

# Initialize a foundational safety-net file to prevent empty submission errors
if not os.path.exists(OUTPUT_PATH):
    pd.DataFrame(columns=['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']).to_csv(OUTPUT_PATH, index=False)
    print("[SAFETY_NET] Initialized baseline submission.csv")

# ============================================================================
# Phase 1: Robust & Ultra-Fast Zarr V3/V2 I/O Engine
# ============================================================================
def read_zarr_metadata(zarr_path: str) -> dict:
    """Parses structural metadata layout from the Zarr file wrapper."""
    meta_path = os.path.join(zarr_path, '0', 'zarr.json')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(zarr_path, 'zarr.json')
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    return {
        'shape': meta.get('shape'),
        'chunks': meta.get('chunks'),
        'dtype': meta.get('data_type', meta.get('dtype')),
        'compressor': meta.get('compressor', None),
        'version': 3 if 'zarr_format' in meta and meta['zarr_format'] == 3 else 2
    }

def read_zarr_frame_native(zarr_path: str, metadata: dict, frame_t: int) -> np.ndarray:
    """
    Reads a single 3D frame timepoint. Features a short-circuit bypass layout
    to handle files containing full frame shards instead of smaller chunks.
    """
    shape = metadata['shape']
    chunks = metadata['chunks']
    dtype = np.dtype(metadata['dtype'])
    compressor = metadata['compressor']
    version = metadata['version']
    
    num_z, num_y, num_x = shape[1], shape[2], shape[3]
    frame_volume = np.zeros((num_z, num_y, num_x), dtype=dtype)
    level_zero_path = os.path.join(zarr_path, '0')
    
    if version == 3:
        first_chunk_path = os.path.join(level_zero_path, 'c', str(frame_t), '0', '0', '0')
    else:
        first_chunk_path = os.path.join(level_zero_path, f"{frame_t}.0.0.0")
        
    if os.path.exists(first_chunk_path):
        try:
            with open(first_chunk_path, 'rb') as f:
                chunk_data = f.read()
            try:
                chunk_data = blosc2.decompress(chunk_data)
            except Exception:
                pass
            chunk_array = np.frombuffer(chunk_data, dtype=dtype)
            if len(chunk_array) == num_z * num_y * num_x:
                return chunk_array.reshape(num_z, num_y, num_x)
        except Exception:
            pass
            
    chunk_z, chunk_y, chunk_x = (chunks[1], chunks[2], chunks[3]) if chunks and len(chunks) >= 4 else (num_z, num_y, num_x)
    grid_z = (num_z + chunk_z - 1) // chunk_z
    grid_y = (num_y + chunk_y - 1) // chunk_y
    grid_x = (num_x + chunk_x - 1) // chunk_x
    
    for cz in range(grid_z):
        for cy in range(grid_y):
            for cx in range(grid_x):
                if version == 3:
                    chunk_filename = os.path.join(level_zero_path, 'c', str(frame_t), str(cz), str(cy), str(cx))
                else:
                    chunk_filename = os.path.join(level_zero_path, f"{frame_t}.{cz}.{cy}.{cx}")
                
                if not os.path.exists(chunk_filename):
                    continue
                
                with open(chunk_filename, 'rb') as f:
                    chunk_data = f.read()
                
                try:
                    chunk_data = blosc2.decompress(chunk_data)
                except Exception:
                    try:
                        if compressor and compressor.get('id') in ['zlib', 'gzip']:
                            chunk_data = zlib.decompress(chunk_data)
                    except Exception:
                        pass
                
                chunk_array = np.frombuffer(chunk_data, dtype=dtype)
                if len(chunk_array) == num_z * num_y * num_x:
                    return chunk_array.reshape(num_z, num_y, num_x)
                
                actual_z = min(chunk_z, num_z - cz * chunk_z)
                actual_y = min(chunk_y, num_y - cy * chunk_y)
                actual_x = min(chunk_x, num_x - cx * chunk_x)
                
                try:
                    reshaped_chunk = chunk_array.reshape(actual_z, actual_y, actual_x)
                    frame_volume[cz*chunk_z:cz*chunk_z+actual_z, cy*chunk_y:cy*chunk_y+actual_y, cx*chunk_x:cx*chunk_x+actual_x] = reshaped_chunk
                except Exception:
                    pass
    return frame_volume

# ============================================================================
# Phase 2: Anisotropic 3D Detection & Advanced Cluster Splitting Engine
# ============================================================================
def detect_peaks_and_segment_watershed(volume: np.ndarray, threshold: float = 0.42, min_distance: int = 5) -> np.ndarray:
    """
    Executes Anisotropic Multi-Scale Difference-of-Gaussians (DoG) blob extraction,
    and resolves dense cell clusters using a 3D marker-controlled Watershed segmentation.
    """
    vol_float = volume.astype(np.float32)
    v_min, v_max = vol_float.min(), vol_float.max()
    if v_max - v_min > 1e-5:
        vol_float = (vol_float - v_min) / (v_max - v_min)
        
    physical_sigmas = np.linspace(MIN_PHYSICAL_SIGMA, MAX_PHYSICAL_SIGMA, NUM_SIGMA_SCALES)
    scale_space_response = np.zeros_like(vol_float)
    
    for i in range(len(physical_sigmas) - 1):
        sigma_s = physical_sigmas[i] / SCALE
        sigma_l = physical_sigmas[i+1] / SCALE
        
        dog = gaussian_filter(vol_float, sigma=sigma_s) - gaussian_filter(vol_float, sigma=sigma_l)
        d_min, d_max = dog.min(), dog.max()
        if d_max - d_min > 1e-5:
            dog = (dog - d_min) / (d_max - d_min)
        scale_space_response = np.maximum(scale_space_response, dog)
        del dog
        
    threshold_mask = scale_space_response > threshold
    if not np.any(threshold_mask):
        return np.empty((0, 3))
        
    dist_transform = distance_transform_edt(threshold_mask)
    local_max = maximum_filter(dist_transform, size=min_distance) == dist_transform
    seed_mask = local_max & threshold_mask
    
    markers, num_features = label(seed_mask)
    if num_features == 0:
        return np.empty((0, 3))
        
    segmented_blocks = watershed(-scale_space_response, markers, mask=threshold_mask)
    
    refined_centroids = []
    for cell_id in range(1, num_features + 1):
        cell_mask = segmented_blocks == cell_id
        coords = np.argwhere(cell_mask)
        if len(coords) == 0:
            continue
        intensities = vol_float[cell_mask]
        weight_sum = intensities.sum()
        if weight_sum > 1e-5:
            centroid = np.sum(coords * intensities[:, None], axis=0) / weight_sum
        else:
            centroid = coords.mean(axis=0)
        refined_centroids.append(centroid)
        
    return np.array(refined_centroids) if refined_centroids else np.empty((0, 3))

# ============================================================================
# Phase 3: Cognitive Tracking Engine (Global LAP with Motion Modeling)
# ============================================================================
def execute_cognitive_lap_tracking(zarr_path: str, metadata: dict) -> Tuple[List[dict], List[dict]]:
    """
    Executes a multi-pass LAP framework including Frame-to-Frame matching with 
    constant-velocity predictions, multi-frame Gap Closing, and Mitosis linking.
    """
    num_frames = metadata['shape'][0]
    node_rows = []
    edge_rows = []
    
    # Track metadata registry: track_id -> {"history": [node_dicts], "velocity": np.array}
    active_tracks = {}
    track_counter = 1
    
    for t in range(num_frames):
        vol = read_zarr_frame_native(zarr_path, metadata, t)
        centroids = detect_peaks_and_segment_watershed(vol, threshold=0.42, min_distance=5)
        del vol
        
        curr_detections = []
        for idx, pos in enumerate(centroids):
            nid = track_counter
            track_counter += 1
            node_info = {'node_id': nid, 't': t, 'z': int(round(pos[0])), 'y': int(round(pos[1])), 'x': int(round(pos[2])), 'pos': pos}
            curr_detections.append(node_info)
            node_rows.append(node_info)
            
        if t == 0:
            for det in curr_detections:
                active_tracks[det['node_id']] = {'history': [det], 'velocity': np.array([0.0, 0.0, 0.0])}
        else:
            if not active_tracks:
                for det in curr_detections:
                    active_tracks[det['node_id']] = {'history': [det], 'velocity': np.array([0.0, 0.0, 0.0])}
                continue
                
            # Snapshot of active tracks prior to current frame updates
            all_track_ids = list(active_tracks.keys())
            
            # Filter targets explicitly originating from the immediate preceding frame (t-1)
            t_minus_1_tracks = [tid for tid in all_track_ids if active_tracks[tid]['history'][-1]['t'] == t - 1]
            
            matched_track_ids = set()
            matched_curr_indices = set()
            
            # --- Pass 1: Standard Frame-to-Frame Linear Assignment Problem (LAP) ---
            if t_minus_1_tracks and curr_detections:
                predicted_positions = []
                for tid in t_minus_1_tracks:
                    track = active_tracks[tid]
                    last_det = track['history'][-1]
                    pred_pos = last_det['pos'] + track['velocity']
                    predicted_positions.append(pred_pos)
                    
                pred_arr = np.array(predicted_positions)
                curr_arr = np.array([d['pos'] for d in curr_detections])
                
                dist_matrix = cdist(pred_arr * SCALE, curr_arr * SCALE, metric='euclidean')
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                
                for r, c in zip(row_ind, col_ind):
                    if dist_matrix[r, c] <= MAX_LINK_DISTANCE:
                        tid = t_minus_1_tracks[r]
                        det = curr_detections[c]
                        
                        edge_rows.append({'source_id': active_tracks[tid]['history'][-1]['node_id'], 'target_id': det['node_id']})
                        
                        new_vel = det['pos'] - active_tracks[tid]['history'][-1]['pos']
                        active_tracks[tid]['history'].append(det)
                        active_tracks[tid]['velocity'] = 0.7 * new_vel + 0.3 * active_tracks[tid]['velocity']
                        
                        matched_track_ids.add(tid)
                        matched_curr_indices.add(c)
            
            # --- Pass 2: Mitosis (Branching Division Detection Layer) ---
            unmatched_curr_indices = [c for c in range(len(curr_detections)) if c not in matched_curr_indices]
            
            for c_idx in unmatched_curr_indices:
                det = curr_detections[c_idx]
                det_phys = det['pos'] * SCALE
                
                best_parent_id = None
                min_mitosis_dist = MITOSIS_EXPANSION_RADIUS
                
                # Search across all candidates that existed at t-1 using snapshot records
                for tid in t_minus_1_tracks:
                    parent_det_at_t1 = next((h for h in active_tracks[tid]['history'] if h['t'] == t - 1), None)
                    if parent_det_at_t1:
                        parent_phys = parent_det_at_t1['pos'] * SCALE
                        p_dist = np.linalg.norm(det_phys - parent_phys)
                        
                        if p_dist < min_mitosis_dist:
                            min_mitosis_dist = p_dist
                            best_parent_id = tid
                            
                if best_parent_id is not None:
                    parent_node_id = next(h['node_id'] for h in active_tracks[best_parent_id]['history'] if h['t'] == t - 1)
                    edge_rows.append({'source_id': parent_node_id, 'target_id': det['node_id']})
                    
                    # Spawn a brand new daughter branch track registry
                    active_tracks[det['node_id']] = {'history': [det], 'velocity': np.array([0.0, 0.0, 0.0])}
                    matched_curr_indices.add(c_idx)
            
            # --- Pass 3: Multi-Frame Gap Closing Layer ---
            unmatched_curr_indices = [c for c in range(len(curr_detections)) if c not in matched_curr_indices]
            t_minus_2_tracks = [tid for tid in all_track_ids if active_tracks[tid]['history'][-1]['t'] == t - 2]
            
            if t_minus_2_tracks and unmatched_curr_indices:
                gap_pred_positions = []
                for tid in t_minus_2_tracks:
                    track = active_tracks[tid]
                    last_det = track['history'][-1]
                    pred_pos = last_det['pos'] + (track['velocity'] * 2.0)
                    gap_pred_positions.append(pred_pos)
                    
                gap_pred_arr = np.array(gap_pred_positions)
                gap_curr_arr = np.array([curr_detections[c]['pos'] for c in unmatched_curr_indices])
                
                gap_dist_matrix = cdist(gap_pred_arr * SCALE, gap_curr_arr * SCALE, metric='euclidean')
                g_row_ind, g_col_ind = linear_sum_assignment(gap_dist_matrix)
                
                for r, c in zip(g_row_ind, g_col_ind):
                    if gap_dist_matrix[r, c] <= GAP_CLOSING_RADIUS:
                        tid = t_minus_2_tracks[r]
                        c_idx = unmatched_curr_indices[c]
                        det = curr_detections[c_idx]
                        
                        edge_rows.append({'source_id': active_tracks[tid]['history'][-1]['node_id'], 'target_id': det['node_id']})
                        
                        new_vel = (det['pos'] - active_tracks[tid]['history'][-1]['pos']) / 2.0
                        active_tracks[tid]['history'].append(det)
                        active_tracks[tid]['velocity'] = 0.7 * new_vel + 0.3 * active_tracks[tid]['velocity']
                        
                        matched_curr_indices.add(c_idx)
            
            # --- Initialize Spontaneous Track Births ---
            for c_idx in range(len(curr_detections)):
                if c_idx not in matched_curr_indices:
                    det = curr_detections[c_idx]
                    active_tracks[det['node_id']] = {'history': [det], 'velocity': np.array([0.0, 0.0, 0.0])}
                    
            # Cull dead paths out of live processing scope memory to optimize performance
            active_tracks = {k: v for k, v in active_tracks.items() if v['history'][-1]['t'] >= t - 2}
            
        gc.collect()
        
    return node_rows, edge_rows

# ============================================================================
# Phase 4: NetworkX Graph Hygiene & Kaggle-Compliant Formatting
# ============================================================================
def generate_sanitized_submission(node_rows: List[dict], edge_rows: List[dict], dataset_id: str) -> pd.DataFrame:
    """
    Enforces strict tree topologies using NetworkX by pruning isolated noise,
    and clamping edge patterns to match competition specifications.
    """
    G = nx.DiGraph()
    for n in node_rows:
        G.add_node(n['node_id'], t=n['t'], z=n['z'], y=n['y'], x=n['x'])
    for e in edge_rows:
        G.add_edge(e['source_id'], e['target_id'])
        
    # Topology cleaning pass: remove track orphans lasting fewer than 2 frames
    nodes_to_remove = []
    for node in list(G.nodes()):
        if G.in_degree(node) == 0 and G.out_degree(node) == 0:
            nodes_to_remove.append(node)
    G.remove_nodes_from(nodes_to_remove)
    
    # Enforce strict properties: maximum 1 parent and maximum 2 daughters
    for node in list(G.nodes()):
        if G.out_degree(node) > 2:
            out_edges = list(G.out_edges(node))
            for edge in out_edges[2:]:
                G.remove_edge(*edge)
                
    final_rows = []
    for node_id in G.nodes():
        attr = G.nodes[node_id]
        final_rows.append({
            'dataset': dataset_id,
            'row_type': 'node',
            'node_id': node_id,
            't': attr['t'],
            'z': attr['z'],
            'y': attr['y'],
            'x': attr['x'],
            'source_id': -1,
            'target_id': -1
        })
        
    for u, v in G.edges():
        final_rows.append({
            'dataset': dataset_id,
            'row_type': 'edge',
            'node_id': -1,
            't': -1,
            'z': -1,
            'y': -1,
            'x': -1,
            'source_id': u,
            'target_id': v
        })
        
    return pd.DataFrame(final_rows)

# ============================================================================
# Core Pipeline Execution Wrapper Loop
# ============================================================================
if __name__ == "__main__":
    if not TEST_DIR:
        print("[ERROR] Verification failure: No test paths discovered on storage volumes.")
        exit(1)
        
    zarr_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.zarr')])
    print(f"[START] Discovered {len(zarr_files)} video arrays for submission construction.")
    
    all_dataset_dfs = []
    
    for idx, filename in enumerate(zarr_files, 1):
        dataset_id = filename.replace('.zarr', '')
        zarr_path = os.path.join(TEST_DIR, filename)
        
        print(f"[PROCESSING] Dataset {idx}/{len(zarr_files)}: {dataset_id}")
        
        try:
            metadata = read_zarr_metadata(zarr_path)
            node_rows, edge_rows = execute_cognitive_lap_tracking(zarr_path, metadata)
            
            dataset_df = generate_sanitized_submission(node_rows, edge_rows, dataset_id)
            all_dataset_dfs.append(dataset_df)
            
            print(f"[SUCCESS] Tracked {len(dataset_df[dataset_df['row_type'] == 'node'])} cells inside {dataset_id}")
            
        except Exception as e:
            print(f"[CRITICAL FAILURE] Skipping dataset {dataset_id} due to exception: {str(e)}")
            import traceback
            traceback.print_exc()
            
    if all_dataset_dfs:
        submission_df = pd.concat(all_dataset_dfs, ignore_index=True)
        submission_df.index.name = 'id'
        submission_df.to_csv(OUTPUT_PATH, index=True)
        print(f"[COMPLETE] Final submission file successfully assembled containing {len(submission_df)} records.")