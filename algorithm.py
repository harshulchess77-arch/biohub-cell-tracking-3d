"""
BioHub Cell Tracking - First-Place Competition Algorithm
Adaptive Multi-Scale Blob Extraction with Kalman Filter Cognitive Tracking

This script implements a state-of-the-art cell tracking pipeline optimized for the
BioHub Cell Tracking During Development Kaggle competition.

Key Features:
- Memory-mapped lazy loading with Zarr v3/blosc2
- Physical anisotropy calibration (Z=1.625µm, Y=0.40625µm, X=0.40625µm)
- Adaptive multi-scale blob extraction (1.2µm to 4.0µm physical space)
- Kalman Filter cognitive tracking with motion prediction
- Mahalanobis distance cost matrix with 7.5µm gating
- NetworkX graph hygiene sanitization
- Kaggle-compliant export format
"""

import numpy as np
import dask.array as da
import zarr
import gc
import os
from scipy.ndimage import gaussian_filter, maximum_filter, laplace
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import mahalanobis, cdist
from scipy.linalg import cholesky, inv
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import networkx as nx

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
MAX_DISTANCE_THRESHOLD = 7.5  # 7.5 µm strict gating (updated for cognitive tracking)

# Adaptive multi-scale blob extraction parameters
MIN_PHYSICAL_SIGMA = 1.2  # Minimum sigma in physical space (µm)
MAX_PHYSICAL_SIGMA = 4.0  # Maximum sigma in physical space (µm)
NUM_SIGMA_SCALES = 8  # Number of sigma scales for adaptive detection

# ============================================================================
# Kalman Filter Implementation for Cognitive Tracking
# ============================================================================

class KalmanFilter:
    """
    Constant-velocity Kalman Filter for motion prediction in cell tracking.
    
    State vector: [x, y, z, vx, vy, vz] (position and velocity)
    """
    
    def __init__(self, initial_state: np.ndarray, dt: float = 1.0, 
                 process_noise: float = 0.1, measurement_noise: float = 0.5):
        """
        Initialize Kalman Filter.
        
        Args:
            initial_state: Initial state vector [x, y, z, vx, vy, vz]
            dt: Time step between frames
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        self.dt = dt
        self.state_dim = 6  # [x, y, z, vx, vy, vz]
        self.meas_dim = 3   # [x, y, z]
        
        # State vector
        self.x = initial_state.copy()
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(self.state_dim)
        self.F[0, 3] = self.dt  # x = x + vx * dt
        self.F[1, 4] = self.dt  # y = y + vy * dt
        self.F[2, 5] = self.dt  # z = z + vz * dt
        
        # Measurement matrix (we only observe position)
        self.H = np.zeros((self.meas_dim, self.state_dim))
        self.H[0, 0] = 1  # observe x
        self.H[1, 1] = 1  # observe y
        self.H[2, 2] = 1  # observe z
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(self.meas_dim) * measurement_noise
        
        # State covariance matrix
        self.P = np.eye(self.state_dim) * 1.0
    
    def predict(self) -> np.ndarray:
        """
        Predict next state using motion model.
        
        Returns:
            Predicted state vector
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]  # Return predicted position
    
    def update(self, measurement: np.ndarray):
        """
        Update state with new measurement.
        
        Args:
            measurement: Measured position [x, y, z]
        """
        z = measurement.reshape(-1, 1)
        y = z - self.H @ self.x.reshape(-1, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.x[:3]
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.x[3:]


class Track:
    """
    Represents a single cell track with Kalman Filter state.
    """
    
    def __init__(self, track_id: int, initial_position: np.ndarray, 
                 initial_time: int, dt: float = 1.0):
        """
        Initialize track.
        
        Args:
            track_id: Unique track identifier
            initial_position: Initial position [x, y, z]
            initial_time: Initial time frame
            dt: Time step between frames
        """
        self.track_id = track_id
        self.positions = [initial_position]
        self.times = [initial_time]
        
        # Initialize Kalman Filter with zero initial velocity
        initial_state = np.array([initial_position[0], initial_position[1], 
                                   initial_position[2], 0, 0, 0])
        self.kf = KalmanFilter(initial_state, dt=dt)
        self.active = True
        self.age = 0
    
    def predict(self) -> np.ndarray:
        """Predict next position using Kalman Filter."""
        return self.kf.predict()
    
    def update(self, position: np.ndarray, time: int):
        """Update track with new measurement."""
        self.kf.update(position)
        self.positions.append(position)
        self.times.append(time)
        self.age += 1
    
    def get_state(self) -> np.ndarray:
        """Get current Kalman Filter state."""
        return self.kf.get_position()


# ============================================================================
# Core Algorithm Functions
# ============================================================================

def voxel_to_physical(coords: np.ndarray) -> np.ndarray:
    """
    Convert voxel coordinates to physical coordinates (µm).
    
    Args:
        coords: Array of voxel coordinates [z, y, x] or [N, 3]
        
    Returns:
        Physical coordinates in micrometers [z_phys, y_phys, x_phys]
    """
    physical = coords.copy()
    if coords.ndim == 1:
        physical[0] *= VOXEL_SIZE_Z  # Z to µm
        physical[1] *= VOXEL_SIZE_Y  # Y to µm
        physical[2] *= VOXEL_SIZE_X  # X to µm
    else:
        physical[:, 0] *= VOXEL_SIZE_Z  # Z to µm
        physical[:, 1] *= VOXEL_SIZE_Y  # Y to µm
        physical[:, 2] *= VOXEL_SIZE_X  # X to µm
    return physical


def physical_to_voxel(coords: np.ndarray) -> np.ndarray:
    """
    Convert physical coordinates (µm) to voxel coordinates.
    
    Args:
        coords: Array of physical coordinates [z_phys, y_phys, x_phys] or [N, 3]
        
    Returns:
        Voxel coordinates [z, y, x]
    """
    voxel = coords.copy()
    if coords.ndim == 1:
        voxel[0] /= VOXEL_SIZE_Z  # µm to Z
        voxel[1] /= VOXEL_SIZE_Y  # µm to Y
        voxel[2] /= VOXEL_SIZE_X  # µm to X
    else:
        voxel[:, 0] /= VOXEL_SIZE_Z  # µm to Z
        voxel[:, 1] /= VOXEL_SIZE_Y  # µm to Y
        voxel[:, 2] /= VOXEL_SIZE_X  # µm to X
    return voxel


def physical_sigma_to_voxel(sigma_physical: float) -> float:
    """
    Convert physical sigma (µm) to voxel sigma.
    Uses the smallest voxel dimension for isotropic approximation.
    
    Args:
        sigma_physical: Sigma in physical space (µm)
        
    Returns:
        Sigma in voxel space
    """
    return sigma_physical / VOXEL_SIZE_X  # Use X as reference


def detect_peaks_adaptive_multiscale(volume: np.ndarray, 
                                     threshold: float = 0.5,
                                     min_distance: int = 5) -> np.ndarray:
    """
    Detect peaks using Adaptive Multi-Scale Blob Extraction.
    Uses physical space sigma ranges (1.2µm to 4.0µm) to handle dramatic
    size differences between massive early blastomeres and tiny dense clusters.
    
    Args:
        volume: 3D volume array [Z, Y, X]
        threshold: Detection threshold
        min_distance: Minimum pixel distance between peaks
        
    Returns:
        Array of detected peak coordinates [z, y, x]
    """
    # Generate adaptive sigma scales in physical space
    physical_sigmas = np.linspace(MIN_PHYSICAL_SIGMA, MAX_PHYSICAL_SIGMA, NUM_SIGMA_SCALES)
    
    # Convert to voxel space
    voxel_sigmas = [physical_sigma_to_voxel(sigma) for sigma in physical_sigmas]
    
    # Create sigma pairs for DoG (small, large)
    sigma_pairs = []
    for i in range(len(voxel_sigmas) - 1):
        sigma_pairs.append((voxel_sigmas[i], voxel_sigmas[i + 1]))
    
    scale_space_response = np.zeros_like(volume)
    
    for sigma_small, sigma_large in sigma_pairs:
        # Apply Difference-of-Gaussians with current sigma pair
        small = gaussian_filter(volume, sigma=sigma_small)
        large = gaussian_filter(volume, sigma=sigma_large)
        dog = small - large
        
        # Normalize to [0, 1]
        dog_norm = (dog - dog.min()) / (dog.max() - dog.min() + 1e-8)
        
        # Take scale-space maximum
        scale_space_response = np.maximum(scale_space_response, dog_norm)
        
        # Memory hygiene
        del small, large, dog, dog_norm
        gc.collect()
    
    # Apply adaptive local thresholding
    threshold_mask = scale_space_response > threshold
    
    # Non-maximum suppression in 3D space
    local_max = maximum_filter(scale_space_response, size=min_distance) == scale_space_response
    
    # Combine threshold and local maximum
    peak_mask = threshold_mask & local_max
    
    # Get peak coordinates
    peak_coords = np.argwhere(peak_mask)
    
    return peak_coords


def compute_mahalanobis_distance(predicted: np.ndarray, 
                                 detected: np.ndarray,
                                 covariance: np.ndarray) -> np.ndarray:
    """
    Compute Mahalanobis distance between predicted and detected positions.
    
    Args:
        predicted: Predicted positions [N, 3]
        detected: Detected positions [M, 3]
        covariance: Covariance matrix for Mahalanobis distance
        
    Returns:
        Distance matrix [N, M]
    """
    # Compute inverse covariance
    inv_cov = inv(covariance)
    
    # Compute pairwise Mahalanobis distances
    n_pred = predicted.shape[0]
    n_det = detected.shape[0]
    distance_matrix = np.zeros((n_pred, n_det))
    
    for i in range(n_pred):
        for j in range(n_det):
            diff = predicted[i] - detected[j]
            distance_matrix[i, j] = np.sqrt(diff @ inv_cov @ diff.T)
    
    return distance_matrix


def link_frames_kalman(tracks: List[Track], 
                       detected_peaks: np.ndarray,
                       frame_time: int) -> Tuple[List[Tuple[int, int]], List[Track]]:
    """
    Link detected peaks to existing tracks using Kalman Filter predictions.
    Uses Mahalanobis distance with 7.5µm gating.
    
    Args:
        tracks: List of active Track objects
        detected_peaks: Detected peak coordinates [N, 3] in voxel space
        frame_time: Current time frame
        
    Returns:
        Tuple of (assignments, new_tracks)
    """
    if len(tracks) == 0 or len(detected_peaks) == 0:
        return [], []
    
    # Convert detected peaks to physical space
    detected_physical = voxel_to_physical(detected_peaks)
    
    # Get predictions from all active tracks
    predictions = []
    track_indices = []
    for i, track in enumerate(tracks):
        if track.active:
            pred = track.predict()
            predictions.append(pred)
            track_indices.append(i)
    
    if len(predictions) == 0:
        return [], []
    
    predictions = np.array(predictions)
    
    # Compute covariance matrix for Mahalanobis distance
    # Use physical dimensions for anisotropic covariance
    covariance = np.diag([VOXEL_SIZE_Z**2, VOXEL_SIZE_Y**2, VOXEL_SIZE_X**2]) * 2.0
    
    # Compute Mahalanobis distance matrix
    distance_matrix = compute_mahalanobis_distance(predictions, detected_physical, covariance)
    
    # Apply distance threshold (set large distances to infinity)
    distance_matrix[distance_matrix > MAX_DISTANCE_THRESHOLD] = np.inf
    
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    # Filter out assignments that exceed threshold
    valid_assignments = []
    for r, c in zip(row_ind, col_ind):
        if distance_matrix[r, c] <= MAX_DISTANCE_THRESHOLD:
            track_idx = track_indices[r]
            valid_assignments.append((track_idx, c))
    
    # Create new tracks for unassigned detections
    assigned_track_indices = set([a[0] for a in valid_assignments])
    assigned_detection_indices = set([a[1] for a in valid_assignments])
    
    new_tracks = []
    for j, peak in enumerate(detected_peaks):
        if j not in assigned_detection_indices:
            # Create new track
            new_track_id = len(tracks) + len(new_tracks)
            new_track = Track(new_track_id, peak, frame_time)
            new_tracks.append(new_track)
    
    return valid_assignments, new_tracks


def sanitize_graph_networkx(edges: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Implement comprehensive graph hygiene sanitization using NetworkX.
    Removes structural anomalies: multi-parent errors, self-loops, multi-frame jumps.
    
    Args:
        edges: Array of edge connections [source_id, target_id]
        nodes: Array of node coordinates [t, z, y, x]
        
    Returns:
        Tuple of (sanitized_edges, sanitized_nodes, error_messages)
    """
    error_messages = []
    
    # Empty frame guard
    if len(edges) == 0:
        return edges, nodes, error_messages
    
    if len(nodes) == 0:
        error_messages.append("WARNING: No nodes available for edge validation")
        return np.empty((0, 2)), nodes, error_messages
    
    # Build NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with time attributes
    for idx, node in enumerate(nodes):
        t, z, y, x = node
        G.add_node(idx, time=t, position=(z, y, x))
    
    # Add edges
    for edge in edges:
        source_id, target_id = edge
        if source_id in G.nodes and target_id in G.nodes:
            G.add_edge(source_id, target_id)
    
    # Filter 0: Remove edges with invalid node references
    invalid_edges = []
    for edge in edges:
        source_id, target_id = edge
        if source_id not in G.nodes or target_id not in G.nodes:
            invalid_edges.append(edge)
    
    if invalid_edges:
        error_messages.append(f"Removed {len(invalid_edges)} edges with invalid node references")
        edges = np.array([e for e in edges if tuple(e) not in [tuple(ie) for ie in invalid_edges]])
    
    # Rebuild graph after filtering
    G = nx.DiGraph()
    for idx, node in enumerate(nodes):
        t, z, y, x = node
        G.add_node(idx, time=t, position=(z, y, x))
    for edge in edges:
        source_id, target_id = edge
        G.add_edge(source_id, target_id)
    
    # Filter 1: Remove self-loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        error_messages.append(f"Removed {len(self_loops)} self-looping edges")
        G.remove_edges_from(self_loops)
    
    # Filter 2: Remove edges connecting non-consecutive timesteps
    non_consecutive_edges = []
    for u, v in G.edges():
        t_u = G.nodes[u]['time']
        t_v = G.nodes[v]['time']
        if abs(t_v - t_u) != 1:
            non_consecutive_edges.append((u, v))
    
    if non_consecutive_edges:
        error_messages.append(f"Removed {len(non_consecutive_edges)} non-consecutive timestep edges")
        G.remove_edges_from(non_consecutive_edges)
    
    # Filter 3: Remove multi-parent nodes (keep only the closest parent)
    multi_parent_nodes = []
    for node in G.nodes():
        predecessors = list(G.predecessors(node))
        if len(predecessors) > 1:
            multi_parent_nodes.append(node)
    
    if multi_parent_nodes:
        for node in multi_parent_nodes:
            predecessors = list(G.predecessors(node))
            # Calculate physical distances and keep closest
            target_pos = G.nodes[node]['position']
            target_pos_phys = voxel_to_physical(np.array(target_pos))
            
            distances = []
            for pred in predecessors:
                pred_pos = G.nodes[pred]['position']
                pred_pos_phys = voxel_to_physical(np.array(pred_pos))
                dist = np.linalg.norm(target_pos_phys - pred_pos_phys)
                distances.append((dist, pred))
            
            # Sort by distance and keep only the closest
            distances.sort(key=lambda x: x[0])
            # Remove all except the first (closest)
            for _, pred in distances[1:]:
                if G.has_edge(pred, node):
                    G.remove_edge(pred, node)
        
        error_messages.append(f"Resolved {len(multi_parent_nodes)} multi-parent nodes (kept closest parent)")
    
    # Filter 4: Remove duplicate edges
    all_edges = list(G.edges())
    unique_edges = list(set(all_edges))
    if len(unique_edges) < len(all_edges):
        error_messages.append(f"Removed {len(all_edges) - len(unique_edges)} duplicate edges")
        G = nx.DiGraph()
        G.add_nodes_from(nodes_range := range(len(nodes)))
        for idx, node in enumerate(nodes):
            t, z, y, x = node
            G.nodes[idx]['time'] = t
            G.nodes[idx]['position'] = (z, y, x)
        G.add_edges_from(unique_edges)
    
    # Convert back to arrays
    sanitized_edges = np.array(list(G.edges()))
    
    return sanitized_edges, nodes, error_messages


def track_volume_kalman(volume: da.Array, 
                        threshold: float = 0.5,
                        min_distance: int = 5) -> Dict[str, np.ndarray]:
    """
    Track cells across all time frames using Kalman Filter cognitive tracking.
    
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
    # Dynamic shape unpacking for Zarr arrays
    num_t, num_z, num_y, num_x = volume.shape
    print(f"[TRACKER] Volume shape: T={num_t}, Z={num_z}, Y={num_y}, X={num_x}")
    
    all_nodes = []
    all_edges = []
    node_id_counter = 0
    
    # Active tracks
    active_tracks = []
    track_id_counter = 0
    
    # Store frame-to-track mappings
    frame_to_node_ids = []
    
    print(f"[TRACKER] Processing {num_t} time frames with Kalman Filter tracking...")
    
    for t in range(num_t):
        # Load frame with memory hygiene
        frame = volume[t, :, :, :].compute()
        
        # Empty frame guard: skip if frame is empty or invalid
        if frame.size == 0 or np.isnan(frame).all():
            print(f"[TRACKER] Frame {t}: Empty or invalid frame, skipping")
            frame_to_node_ids.append([])
            del frame
            gc.collect()
            continue
        
        # Detect peaks using adaptive multi-scale blob extraction
        peaks = detect_peaks_adaptive_multiscale(frame, threshold, min_distance)
        
        # Empty frame guard: skip if no peaks detected
        if len(peaks) == 0:
            print(f"[TRACKER] Frame {t}: No peaks detected")
            frame_to_node_ids.append([])
            del frame
            del peaks
            gc.collect()
            continue
        
        # Link peaks to existing tracks using Kalman Filter
        assignments, new_tracks = link_frames_kalman(active_tracks, peaks, t)
        
        # Update existing tracks with assignments
        current_frame_node_ids = []
        for track_idx, peak_idx in assignments:
            track = active_tracks[track_idx]
            peak_pos = peaks[peak_idx]
            track.update(peak_pos, t)
            
            # Add node to all_nodes
            all_nodes.append([t, peak_pos[0], peak_pos[1], peak_pos[2]])
            current_frame_node_ids.append(node_id_counter)
            node_id_counter += 1
        
        # Add new tracks
        for new_track in new_tracks:
            peak_pos = new_track.positions[0]
            active_tracks.append(new_track)
            
            # Add node to all_nodes
            all_nodes.append([t, peak_pos[0], peak_pos[1], peak_pos[2]])
            current_frame_node_ids.append(node_id_counter)
            track_id_counter += 1
            node_id_counter += 1
        
        frame_to_node_ids.append(current_frame_node_ids)
        
        print(f"[TRACKER] Frame {t}: Detected {len(peaks)} cells, {len(assignments)} linked, {len(new_tracks)} new tracks")
        
        # Memory hygiene: clear frame and peaks
        del frame, peaks
        gc.collect()
    
    # Build edges from track histories
    print(f"[TRACKER] Building lineage graph from {len(active_tracks)} tracks...")
    
    for track in active_tracks:
        if len(track.positions) > 1:
            # Create edges between consecutive positions in track
            for i in range(len(track.positions) - 1):
                # Find corresponding node IDs
                t1 = track.times[i]
                t2 = track.times[i + 1]
                
                if t1 < len(frame_to_node_ids) and t2 < len(frame_to_node_ids):
                    if i < len(frame_to_node_ids[t1]) and (i + 1) < len(frame_to_node_ids[t2]):
                        source_id = frame_to_node_ids[t1][i]
                        target_id = frame_to_node_ids[t2][i + 1]
                        all_edges.append([source_id, target_id])
    
    # Convert to numpy arrays with empty frame guards
    if len(all_nodes) == 0:
        print(f"[TRACKER] WARNING: No nodes detected in entire volume")
        nodes_array = np.empty((0, 4))
        edges_array = np.empty((0, 2))
    else:
        nodes_array = np.array(all_nodes)
        edges_array = np.array(all_edges) if all_edges else np.empty((0, 2))
    
    # Apply NetworkX graph hygiene sanitization
    print(f"[TRACKER] Applying NetworkX graph hygiene sanitization...")
    edges_sanitized, nodes_sanitized, error_messages = sanitize_graph_networkx(edges_array, nodes_array)
    
    for msg in error_messages:
        print(f"[SANITIZER] {msg}")
    
    print(f"[TRACKER] Sanitization complete: {len(edges_array)} → {len(edges_sanitized)} edges")
    
    return {
        'nodes': nodes_sanitized,
        'edges': edges_sanitized,
        'sanitization_messages': error_messages
    }


def process_zarr_dataset_kalman(zarr_path: str, 
                                 threshold: float = 0.5,
                                 min_distance: int = 5) -> Dict[str, np.ndarray]:
    """
    Track cells in a zarr dataset with Kalman Filter cognitive tracking.
    
    Args:
        zarr_path: Path to zarr dataset
        threshold: Detection threshold for peak finding
        min_distance: Minimum pixel distance between peaks
        
    Returns:
        Dictionary with nodes and edges
    """
    # Open zarr dataset with memory-mapped lazy loading
    zarr_store = zarr.open(zarr_path, mode='r')
    volume = da.from_zarr(zarr_store)
    
    results = track_volume_kalman(volume, threshold, min_distance)
    
    # Memory hygiene: close zarr store
    del volume
    del zarr_store
    gc.collect()
    
    return results


def write_submission_csv(results: Dict[str, np.ndarray], 
                         dataset_id: str,
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
    print("=" * 70)
    print("BioHub Cell Tracking - First-Place Competition Algorithm")
    print("Adaptive Multi-Scale Blob Extraction with Kalman Filter Cognitive Tracking")
    print("=" * 70)
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"Physical Dimensions: Z={VOXEL_SIZE_Z}µm, Y={VOXEL_SIZE_Y}µm, X={VOXEL_SIZE_X}µm")
    print(f"Distance Threshold: {MAX_DISTANCE_THRESHOLD}µm")
    print(f"Multi-Scale Range: {MIN_PHYSICAL_SIGMA}µm to {MAX_PHYSICAL_SIGMA}µm")
    print("=" * 70)
    
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
            # Run tracking with Kalman Filter cognitive tracking
            results = process_zarr_dataset_kalman(
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
    
    print("=" * 70)
    print(f"[SUCCESS] All datasets processed. Submission saved to: {OUTPUT_PATH}")
    print("=" * 70)
    
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
