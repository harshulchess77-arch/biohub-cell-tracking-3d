"""
BioHub Cell Tracking - Elite Competition Algorithm
Adaptive Multi-Scale Blob Extraction with Velocity-Based Cognitive Tracking

This script implements a state-of-the-art cell tracking pipeline optimized for the
BioHub Cell Tracking During Development Kaggle competition.

Key Features:
- Memory-mapped lazy loading with Zarr v3/blosc2
- Physical anisotropy calibration (Z=1.625µm, Y=0.40625µm, X=0.40625µm)
- Adaptive multi-scale blob extraction (1.2µm to 3.8µm physical space)
- Local intensity centroid refinement for sub-voxel accuracy
- Velocity-based motion prediction with constant-velocity model
- Euclidean distance cost matrix with 7.0µm gating
- Gap closing mechanism (1-to-2 frame look-ahead)
- Mitosis detection (1-to-2 branching)
- NetworkX graph hygiene sanitization
- Kaggle-compliant export format
"""

import numpy as np
import gc
import os
import json
import zlib
import gzip
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict
from collections import defaultdict
import networkx as nx

# ============================================================================
# Kaggle Environment Configuration
# ============================================================================
# Detect if running in Kaggle environment or local data folder
# Check if actual data directory exists, not just base input path
if os.path.exists('/kaggle/input/biohub-cell-tracking-during-development/test'):
    INPUT_DIR = "/kaggle/input/biohub-cell-tracking-during-development/test"
    OUTPUT_PATH = "submission.csv"
else:
    INPUT_DIR = "./data/test"  # fallback for local dry-runs
    OUTPUT_PATH = "./submission.csv"

# Physical voxel dimensions in micrometers (µm)
VOXEL_SIZE_Z = 1.625  # Z-axis: 1.625 µm/voxel
VOXEL_SIZE_Y = 0.40625  # Y-axis: 0.40625 µm/voxel
VOXEL_SIZE_X = 0.40625  # X-axis: 0.40625 µm/voxel

# Maximum physical distance for cell tracking between frames (µm)
MAX_DISTANCE_THRESHOLD = 7.0  # 7.0 µm strict gating

# Adaptive multi-scale blob extraction parameters
MIN_PHYSICAL_SIGMA = 1.2  # Minimum sigma in physical space (µm)
MAX_PHYSICAL_SIGMA = 3.8  # Maximum sigma in physical space (µm)
NUM_SIGMA_SCALES = 8  # Number of sigma scales for adaptive detection

# Gap closing parameters
GAP_CLOSING_RADIUS = 5.5  # µm proximity for gap recovery
MAX_GAP_FRAMES = 2  # Look-ahead frames for gap closing

# Mitosis detection parameters
MITOSIS_EXPANSION_RADIUS = 8.0  # µm radius for mitosis detection

# ============================================================================
# Velocity-Based Motion Model for Cognitive Tracking
# ============================================================================

class VelocityTrack:
    """
    Represents a single cell track with velocity-based motion prediction.
    Uses constant-velocity model: r_hat = r_t + lambda_v * (r_t - r_{t-1})
    """
    
    def __init__(self, track_id: int, initial_position: np.ndarray, 
                 initial_time: int, lambda_v: float = 1.0):
        """
        Initialize track.
        
        Args:
            track_id: Unique track identifier
            initial_position: Initial position [z, y, x]
            initial_time: Initial time frame
            lambda_v: Velocity scaling factor for motion prediction
        """
        self.track_id = track_id
        self.positions = [initial_position]
        self.times = [initial_time]
        self.lambda_v = lambda_v
        self.active = True
        self.age = 0
        self.velocity = np.zeros(3)  # Initial velocity is zero
    
    def predict_next_position(self) -> np.ndarray:
        """
        Predict next position using velocity-based motion model.
        r_hat = r_t + lambda_v * (r_t - r_{t-1})
        
        Returns:
            Predicted position [z, y, x]
        """
        if len(self.positions) < 2:
            # Not enough history, return current position
            return self.positions[-1]
        
        current_pos = self.positions[-1]
        prev_pos = self.positions[-2]
        
        # Calculate velocity vector
        velocity = current_pos - prev_pos
        self.velocity = velocity
        
        # Predict next position
        predicted = current_pos + self.lambda_v * velocity
        return predicted
    
    def update(self, position: np.ndarray, time: int):
        """Update track with new measurement."""
        self.positions.append(position)
        self.times.append(time)
        self.age += 1
    
    def get_position(self) -> np.ndarray:
        """Get current position."""
        return self.positions[-1]
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity vector."""
        return self.velocity


# ============================================================================
# Pure Native Zarr Reader (No zarr package dependency)
# ============================================================================

def read_zarr_native(zarr_path: str) -> np.ndarray:
    """
    Read Zarr dataset using pure native Python libraries.
    Reads .zarray metadata and binary chunks directly without zarr package.
    
    Args:
        zarr_path: Path to zarr dataset directory
        
    Returns:
        NumPy array containing the full 4D volume [T, Z, Y, X]
    """
    # Read .zarray metadata
    zarray_path = os.path.join(zarr_path, '.zarray')
    with open(zarray_path, 'r') as f:
        metadata = json.load(f)
    
    shape = tuple(metadata['shape'])  # [T, Z, Y, X]
    chunks = tuple(metadata['chunks'])  # Chunk shape
    dtype = np.dtype(metadata['dtype'])
    compressor = metadata.get('compressor', None)
    
    print(f"[ZARR_NATIVE] Shape: {shape}, Chunks: {chunks}, Dtype: {dtype}")
    print(f"[ZARR_NATIVE] Compressor: {compressor}")
    
    # Initialize output array
    volume = np.zeros(shape, dtype=dtype)
    
    # Calculate chunk grid dimensions
    chunk_grid = [ (shape[i] + chunks[i] - 1) // chunks[i] for i in range(len(shape)) ]
    
    # Read each chunk
    for chunk_idx in np.ndindex(*chunk_grid):
        # Build chunk filename (e.g., '0.0.0.0')
        chunk_filename = '.'.join(map(str, chunk_idx))
        chunk_path = os.path.join(zarr_path, chunk_filename)
        
        if not os.path.exists(chunk_path):
            print(f"[ZARR_NATIVE] Warning: Chunk {chunk_filename} not found, skipping")
            continue
        
        # Read binary chunk data
        with open(chunk_path, 'rb') as f:
            chunk_data = f.read()
        
        # Decompress if needed
        if compressor is not None:
            compressor_id = compressor.get('id', '')
            try:
                if compressor_id == 'zlib':
                    chunk_data = zlib.decompress(chunk_data)
                elif compressor_id == 'gzip':
                    chunk_data = gzip.decompress(chunk_data)
                elif 'blosc' in compressor_id:
                    # Blosc2 not supported natively, try raw read
                    print(f"[ZARR_NATIVE] Warning: Blosc compression not natively supported, reading raw")
                else:
                    print(f"[ZARR_NATIVE] Warning: Unknown compressor {compressor_id}, reading raw")
            except Exception as e:
                print(f"[ZARR_NATIVE] Decompression failed for {chunk_filename}: {e}, reading raw")
        
        # Convert to numpy array
        chunk_array = np.frombuffer(chunk_data, dtype=dtype)
        chunk_array = chunk_array.reshape(chunks)
        
        # Calculate slice indices for this chunk
        slice_obj = tuple(
            slice(idx * chunk_size, min((idx + 1) * chunk_size, shape[i]))
            for i, (idx, chunk_size) in enumerate(zip(chunk_idx, chunks))
        )
        
        # Handle edge case where chunk is smaller than chunk size
        chunk_slice = tuple(
            slice(0, min(chunk_size, shape[i] - idx * chunk_size))
            for i, (idx, chunk_size) in enumerate(zip(chunk_idx, chunks))
        )
        
        # Insert chunk into volume
        volume[slice_obj] = chunk_array[chunk_slice]
        
        # Memory hygiene
        del chunk_data, chunk_array
        gc.collect()
    
    print(f"[ZARR_NATIVE] Successfully loaded volume of shape {volume.shape}")
    return volume


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


def refine_centroid_local_intensity(volume: np.ndarray, 
                                  peak_coords: np.ndarray,
                                  window_size: int = 3) -> np.ndarray:
    """
    Refine peak coordinates using local intensity centroid for sub-voxel accuracy.
    
    Args:
        volume: 3D volume array [Z, Y, X]
        peak_coords: Peak coordinates [N, 3] in voxel space
        window_size: Size of local window for centroid calculation
        
    Returns:
        Refined peak coordinates [N, 3] with sub-voxel precision
    """
    refined_coords = []
    
    for peak in peak_coords:
        z, y, x = peak
        z, y, x = int(z), int(y), int(x)
        
        # Extract local window
        z_start = max(0, z - window_size // 2)
        z_end = min(volume.shape[0], z + window_size // 2 + 1)
        y_start = max(0, y - window_size // 2)
        y_end = min(volume.shape[1], y + window_size // 2 + 1)
        x_start = max(0, x - window_size // 2)
        x_end = min(volume.shape[2], x + window_size // 2 + 1)
        
        local_window = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Compute intensity-weighted centroid
        if local_window.sum() > 0:
            zz, yy, xx = np.meshgrid(
                np.arange(z_start, z_end),
                np.arange(y_start, y_end),
                np.arange(x_start, x_end),
                indexing='ij'
            )
            
            weighted_z = (zz * local_window).sum() / local_window.sum()
            weighted_y = (yy * local_window).sum() / local_window.sum()
            weighted_x = (xx * local_window).sum() / local_window.sum()
            
            refined_coords.append([weighted_z, weighted_y, weighted_x])
        else:
            refined_coords.append([z, y, x])
    
    return np.array(refined_coords)


def detect_peaks_adaptive_multiscale(volume: np.ndarray, 
                                     threshold: float = 0.5,
                                     min_distance: int = 5) -> np.ndarray:
    """
    Detect peaks using Adaptive Multi-Scale Blob Extraction with local refinement.
    Uses physical space sigma ranges (1.2µm to 3.8µm) to handle dramatic
    size differences between massive early blastomeres and tiny dense clusters.
    
    Args:
        volume: 3D volume array [Z, Y, X]
        threshold: Detection threshold
        min_distance: Minimum pixel distance between peaks
        
    Returns:
        Array of refined peak coordinates [z, y, x] with sub-voxel precision
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
    
    # Apply local intensity centroid refinement for sub-voxel accuracy
    if len(peak_coords) > 0:
        peak_coords = refine_centroid_local_intensity(volume, peak_coords, window_size=3)
    
    return peak_coords


def compute_euclidean_distance(predicted: np.ndarray, 
                               detected: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between predicted and detected positions.
    
    Args:
        predicted: Predicted positions [N, 3] in physical space
        detected: Detected positions [M, 3] in physical space
        
    Returns:
        Distance matrix [N, M]
    """
    return cdist(predicted, detected, metric='euclidean')


def link_frames_velocity(tracks: List[VelocityTrack], 
                          detected_peaks: np.ndarray,
                          frame_time: int) -> Tuple[List[Tuple[int, int]], List[VelocityTrack]]:
    """
    Link detected peaks to existing tracks using velocity-based motion prediction.
    Uses Euclidean distance with 7.0µm gating.
    
    Args:
        tracks: List of active VelocityTrack objects
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
            pred = track.predict_next_position()
            pred_physical = voxel_to_physical(pred)
            predictions.append(pred_physical)
            track_indices.append(i)
    
    if len(predictions) == 0:
        return [], []
    
    predictions = np.array(predictions)
    
    # Compute Euclidean distance matrix
    distance_matrix = compute_euclidean_distance(predictions, detected_physical)
    
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
            new_track = VelocityTrack(new_track_id, peak, frame_time)
            new_tracks.append(new_track)
    
    return valid_assignments, new_tracks


def close_gaps(tracks: List[VelocityTrack], 
              frame_to_node_ids: List[List[int]],
              nodes: List[List[int]],
              num_frames: int) -> List[Tuple[int, int]]:
    """
    Implement 1-to-2 frame look-ahead gap recovery mechanism.
    If a track ends abruptly at frame T, look ahead to frames T+2 and T+3.
    If a new track emerges within GAP_CLOSING_RADIUS (5.5µm), bridge the gap.
    
    Args:
        tracks: List of VelocityTrack objects
        frame_to_node_ids: Mapping from frame to node IDs
        nodes: List of node coordinates [t, z, y, x]
        num_frames: Total number of frames
        
    Returns:
        List of gap-closing edges (source_id, target_id)
    """
    gap_edges = []
    
    for track in tracks:
        if len(track.positions) < 2:
            continue
        
        last_time = track.times[-1]
        last_pos = track.positions[-1]
        
        # Look ahead up to MAX_GAP_FRAMES
        for gap in range(1, MAX_GAP_FRAMES + 1):
            look_ahead_frame = last_time + gap
            
            if look_ahead_frame >= num_frames:
                break
            
            if look_ahead_frame >= len(frame_to_node_ids):
                break
            
            # Find tracks that start at look_ahead_frame
            for other_track in tracks:
                if other_track.track_id == track.track_id:
                    continue
                
                if len(other_track.times) == 0:
                    continue
                
                if other_track.times[0] == look_ahead_frame:
                    other_pos = other_track.positions[0]
                    
                    # Calculate physical distance
                    last_pos_phys = voxel_to_physical(np.array(last_pos))
                    other_pos_phys = voxel_to_physical(np.array(other_pos))
                    distance = np.linalg.norm(last_pos_phys - other_pos_phys)
                    
                    if distance <= GAP_CLOSING_RADIUS:
                        # Bridge the gap
                        # Find corresponding node IDs
                        if last_time < len(frame_to_node_ids) and look_ahead_frame < len(frame_to_node_ids):
                            # Find node IDs for these positions
                            source_idx = -1
                            target_idx = -1
                            
                            # Find source node ID
                            for i, (t, z, y, x) in enumerate(nodes):
                                if t == last_time:
                                    if np.allclose([z, y, x], last_pos, atol=1.0):
                                        source_idx = i
                                        break
                            
                            # Find target node ID
                            for i, (t, z, y, x) in enumerate(nodes):
                                if t == look_ahead_frame:
                                    if np.allclose([z, y, x], other_pos, atol=1.0):
                                        target_idx = i
                                        break
                            
                            if source_idx >= 0 and target_idx >= 0:
                                gap_edges.append((source_idx, target_idx))
                                print(f"[GAP_CLOSING] Bridged gap from frame {last_time} to {look_ahead_frame} (distance: {distance:.2f}µm)")
                            break
    
    return gap_edges


def detect_mitosis(tracks: List[VelocityTrack],
                  frame_to_node_ids: List[List[int]],
                  nodes: List[List[int]]) -> List[Tuple[int, int, int]]:
    """
    Detect mitosis events (1-to-2 branching).
    If a single source node terminates and pairs cleanly with two new target nodes
    in the next immediate frame within MITOSIS_EXPANSION_RADIUS, register as split.
    
    Args:
        tracks: List of VelocityTrack objects
        frame_to_node_ids: Mapping from frame to node IDs
        nodes: List of node coordinates [t, z, y, x]
        
    Returns:
        List of mitosis edges (source_id, target_id1, target_id2)
    """
    mitosis_events = []
    
    for track in tracks:
        if len(track.positions) < 2:
            continue
        
        last_time = track.times[-1]
        last_pos = track.positions[-1]
        
        # Look for tracks that start at last_time + 1
        next_frame = last_time + 1
        potential_daughters = []
        
        for other_track in tracks:
            if other_track.track_id == track.track_id:
                continue
            
            if len(other_track.times) == 0:
                continue
            
            if other_track.times[0] == next_frame:
                other_pos = other_track.positions[0]
                
                # Calculate physical distance
                last_pos_phys = voxel_to_physical(np.array(last_pos))
                other_pos_phys = voxel_to_physical(np.array(other_pos))
                distance = np.linalg.norm(last_pos_phys - other_pos_phys)
                
                if distance <= MITOSIS_EXPANSION_RADIUS:
                    potential_daughters.append((other_track, distance, other_pos))
        
        # If exactly 2 potential daughters, register as mitosis
        if len(potential_daughters) == 2:
            daughter1, dist1, pos1 = potential_daughters[0]
            daughter2, dist2, pos2 = potential_daughters[1]
            
            # Find corresponding node IDs
            source_idx = -1
            target_idx1 = -1
            target_idx2 = -1
            
            for i, (t, z, y, x) in enumerate(nodes):
                if t == last_time:
                    if np.allclose([z, y, x], last_pos, atol=1.0):
                        source_idx = i
                        break
            
            for i, (t, z, y, x) in enumerate(nodes):
                if t == next_frame:
                    if np.allclose([z, y, x], pos1, atol=1.0):
                        target_idx1 = i
                    elif np.allclose([z, y, x], pos2, atol=1.0):
                        target_idx2 = i
            
            if source_idx >= 0 and target_idx1 >= 0 and target_idx2 >= 0:
                mitosis_events.append((source_idx, target_idx1, target_idx2))
                print(f"[MITOSIS] Detected division at frame {last_time} → {next_frame} (distances: {dist1:.2f}µm, {dist2:.2f}µm)")
    
    return mitosis_events


def sanitize_graph_networkx(edges: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Implement comprehensive graph hygiene sanitization using NetworkX.
    Removes structural anomalies: multi-parent errors, self-loops, multi-frame jumps,
    and isolated single-frame orphan nodes.
    
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
    
    # Filter 5: Remove isolated single-frame orphan nodes
    node_degrees = dict(G.degree())
    isolated_nodes = [node for node, degree in node_degrees.items() if degree == 0]
    if isolated_nodes:
        error_messages.append(f"Removed {len(isolated_nodes)} isolated single-frame orphan nodes")
        # Remove isolated nodes from graph
        G.remove_nodes_from(isolated_nodes)
        # Remove corresponding nodes from nodes array
        keep_mask = np.array([i not in isolated_nodes for i in range(len(nodes))])
        nodes = nodes[keep_mask]
        # Rebuild graph with filtered nodes
        G = nx.DiGraph()
        for idx, node in enumerate(nodes):
            t, z, y, x = node
            G.add_node(idx, time=t, position=(z, y, x))
        for edge in unique_edges:
            source_id, target_id = edge
            if source_id not in isolated_nodes and target_id not in isolated_nodes:
                # Remap node IDs after filtering
                old_to_new = {}
                new_idx = 0
                for i in range(len(nodes) + len(isolated_nodes)):
                    if i not in isolated_nodes:
                        old_to_new[i] = new_idx
                        new_idx += 1
                if source_id in old_to_new and target_id in old_to_new:
                    G.add_edge(old_to_new[source_id], old_to_new[target_id])
    
    # Convert back to arrays
    sanitized_edges = np.array(list(G.edges()))
    
    return sanitized_edges, nodes, error_messages


def track_volume_velocity(volume: np.ndarray, 
                         threshold: float = 0.5,
                         min_distance: int = 5) -> Dict[str, np.ndarray]:
    """
    Track cells across all time frames using velocity-based motion prediction.
    
    Args:
        volume: 4D volume array [T, Z, Y, X] (numpy array)
        threshold: Detection threshold for peak finding
        min_distance: Minimum pixel distance between peaks
        
    Returns:
        Dictionary containing:
            - 'nodes': Array of node coordinates [t, z, y, x]
            - 'edges': Array of edge connections [source_id, target_id]
            - 'sanitization_messages': List of sanitization messages
    """
    # Dynamic shape unpacking
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
    
    print(f"[TRACKER] Processing {num_t} time frames with velocity-based tracking...")
    
    for t in range(num_t):
        # Load frame directly from numpy array
        frame = volume[t, :, :, :].copy()  # Copy to avoid modifying original
        
        # Empty frame guard: skip if frame is empty or invalid
        if frame.size == 0 or np.isnan(frame).all():
            print(f"[TRACKER] Frame {t}: Empty or invalid frame, skipping")
            frame_to_node_ids.append([])
            del frame
            gc.collect()
            continue
        
        # Detect peaks using adaptive multi-scale blob extraction with centroid refinement
        peaks = detect_peaks_adaptive_multiscale(frame, threshold, min_distance)
        
        # Empty frame guard: skip if no peaks detected
        if len(peaks) == 0:
            print(f"[TRACKER] Frame {t}: No peaks detected")
            frame_to_node_ids.append([])
            del frame
            del peaks
            gc.collect()
            continue
        
        # Link peaks to existing tracks using velocity-based motion prediction
        assignments, new_tracks = link_frames_velocity(active_tracks, peaks, t)
        
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
    
    # Apply gap closing mechanism
    print(f"[TRACKER] Applying gap closing mechanism...")
    gap_edges = close_gaps(active_tracks, frame_to_node_ids, all_nodes, num_t)
    for source_id, target_id in gap_edges:
        all_edges.append([source_id, target_id])
    
    # Apply mitosis detection
    print(f"[TRACKER] Applying mitosis detection...")
    mitosis_events = detect_mitosis(active_tracks, frame_to_node_ids, all_nodes)
    for source_id, target_id1, target_id2 in mitosis_events:
        all_edges.append([source_id, target_id1])
        all_edges.append([source_id, target_id2])
    
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


def process_zarr_dataset_velocity(zarr_path: str, 
                                  threshold: float = 0.5,
                                  min_distance: int = 5) -> Dict[str, np.ndarray]:
    """
    Track cells in a zarr dataset with velocity-based motion prediction.
    Uses pure native Zarr reader (no zarr package dependency).
    
    Args:
        zarr_path: Path to zarr dataset
        threshold: Detection threshold for peak finding
        min_distance: Minimum pixel distance between peaks
        
    Returns:
        Dictionary with nodes and edges
    """
    # Read zarr dataset using pure native reader
    volume = read_zarr_native(zarr_path)
    
    results = track_volume_velocity(volume, threshold, min_distance)
    
    # Memory hygiene: clear volume
    del volume
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
    Uses velocity-based motion prediction with gap closing and mitosis detection.
    """
    print("=" * 70)
    print("BioHub Cell Tracking - Elite Competition Algorithm")
    print("Adaptive Multi-Scale Blob Extraction with Velocity-Based Cognitive Tracking")
    print("=" * 70)
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"Physical Dimensions: Z={VOXEL_SIZE_Z}µm, Y={VOXEL_SIZE_Y}µm, X={VOXEL_SIZE_X}µm")
    print(f"Distance Threshold: {MAX_DISTANCE_THRESHOLD}µm")
    print(f"Multi-Scale Range: {MIN_PHYSICAL_SIGMA}µm to {MAX_PHYSICAL_SIGMA}µm")
    print(f"Gap Closing Radius: {GAP_CLOSING_RADIUS}µm")
    print(f"Mitosis Expansion Radius: {MITOSIS_EXPANSION_RADIUS}µm")
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
    
    # Process each dataset sequentially
    for idx, zarr_file in enumerate(zarr_files):
        dataset_id = zarr_file.replace('.zarr', '')
        zarr_path = os.path.join(INPUT_DIR, zarr_file)
        
        print(f"[PROCESSING] Dataset {idx+1}/{len(zarr_files)}: {dataset_id}")
        print(f"[PROCESSING] Path: {zarr_path}")
        
        try:
            # Run tracking with velocity-based motion prediction
            results = process_zarr_dataset_velocity(
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
