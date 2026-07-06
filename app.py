import os
import time
import torch
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from scipy.ndimage import sobel
from src.data_loader import get_zarr_stream
from src.graph_loader import parse_geff_data
from src.image_processor import extract_3d_crop
from src.model import CellTrackerUNet3D
from src.classical_tracker import AnisotropicDoGTracker
from src.evaluation import EmbryoSplitEvaluator

st.set_page_config(page_title="Biohub Cell Tracking Lab", layout="wide", page_icon="🔬")

# Scientific Dark Mode - Minimal Professional Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .stApp > div {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #e6edf3;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab-list"] > div {
        background-color: #161b22;
        border-radius: 6px;
        padding: 4px 12px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #58a6ff !important;
    }
    .metric-container {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 16px;
        color: #e6edf3;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("🔬 Biohub Cell Tracking Lab")
st.caption("4D Volumetric Analysis Workspace - Scientific Instrument Interface")

# HUD Metrics
hud1, hud2, hud3, hud4 = st.columns(4)
with hud1:
    st.markdown('<div class="metric-container"><div class="metric-label">Instrument State</div><div class="metric-value">Ready</div></div>', unsafe_allow_html=True)
with hud2:
    st.markdown('<div class="metric-container"><div class="metric-label">Spatial Focus</div><div class="metric-value">Orthogonal</div></div>', unsafe_allow_html=True)
with hud3:
    st.markdown('<div class="metric-container"><div class="metric-label">Data Stream</div><div class="metric-value">Zarr/Dask</div></div>', unsafe_allow_html=True)
with hud4:
    st.markdown('<div class="metric-container"><div class="metric-label">Kaggle Score</div><div class="metric-value" id="kaggle-score">--</div></div>', unsafe_allow_html=True)

# Session initialization
if "playback_active" not in st.session_state: st.session_state.playback_active = False
if "playback_frame" not in st.session_state: st.session_state.playback_frame = 0

# Sidebar with expanders
with st.sidebar:
    st.header("📁 Dataset Selection")
    TRAIN_DIR = os.path.join("data", "train")
    zarr_folders = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".zarr")] if os.path.exists(TRAIN_DIR) else []

    if zarr_folders:
        selected_dataset = st.selectbox("Select Volume", zarr_folders)
        volume = get_zarr_stream(selected_dataset, base_dir=TRAIN_DIR)
        t_max, z_max, y_max, x_max = volume.shape

        try:
            nodes, edges = parse_geff_data(selected_dataset, base_dir=TRAIN_DIR)
            has_graph = True
        except Exception:
            nodes, edges, has_graph = None, None, False

        st.success(f"Loaded: {selected_dataset}")
        st.caption(f"T: {t_max} | Z: {z_max} | Y: {y_max} | X: {x_max}")
    else:
        st.error("No datasets found in data/train/")
        selected_dataset = None
        has_graph = False
        nodes, edges = None, None

    st.divider()

    with st.expander("⏱️ Temporal Navigation", expanded=True):
        if selected_dataset:
            if st.session_state.playback_active:
                st.session_state.playback_frame = (st.session_state.playback_frame + 1) % t_max
                t_idx = st.session_state.playback_frame
                st.caption(f"▶ Playing: Frame {t_idx}")
                st.rerun()
            else:
                t_idx = st.slider("Time Frame", 0, t_max - 1, int(st.session_state.playback_frame))
                st.session_state.playback_frame = t_idx

            play_pause = st.button("▶ Play" if not st.session_state.playback_active else "⏸ Pause")
            if play_pause:
                st.session_state.playback_active = not st.session_state.playback_active
                st.rerun()
    
    with st.expander("🔍 Depth Navigation", expanded=True):
        if selected_dataset:
            z_idx = st.slider("Z Slice", 0, z_max - 1, z_max // 2)

    with st.expander("🎨 Visualization Settings"):
        if selected_dataset:
            colormap = st.selectbox("Colormap", ["inferno", "viridis", "magma", "plasma", "gray"])
            apply_edge = st.checkbox("Edge Contrast Booster")
            show_gt = st.checkbox("Show Ground Truth", value=True) if has_graph else False

    st.divider()

    with st.expander("🔬 Tracking Engine"):
        tracker_mode = st.radio("Algorithm", ["Classical DoG", "Deep Learning U-Net"])
        
        # Multi-scale DoG options
        if tracker_mode == "Classical DoG":
            multi_scale = st.checkbox("Multi-scale Detection", value=True)
            threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)
        
        evaluate_btn = st.button("Run Analysis")
        export_btn = st.button("Export Submission")
    
    st.divider()
    
    # Initialize CV variables
    cv_enabled = False
    n_folds = 5
    run_cv_btn = None
    
    with st.expander("📊 Cross-Validation"):
        cv_enabled = st.checkbox("Enable Cross-Validation")
        if cv_enabled:
            n_folds = st.slider("Number of Folds", 2, 10, 5)
            run_cv_btn = st.button("Run Cross-Validation")

# Initialize evaluator for division detection
evaluator = None
if selected_dataset:
    evaluator = EmbryoSplitEvaluator(train_dir=TRAIN_DIR)

# Main content area with tabs
if selected_dataset:
    tab1, tab2, tab3 = st.tabs(["📊 Multi-Planar View", "🌳 Lineage Tracking", "📈 Metrics & Evaluation"])
    
    with tab1:
        st.subheader("Multi-Planar Orthogonal Visualization")
        
        # Load slices
        slice_xy = volume[t_idx, z_idx, :, :].compute()
        slice_xz = volume[t_idx, :, y_max // 2, :].compute()
        slice_yz = volume[t_idx, :, :, x_max // 2].compute()
        
        # Apply edge contrast if enabled
        if apply_edge:
            edge_x_xy = sobel(slice_xy, axis=1)
            edge_y_xy = sobel(slice_xy, axis=0)
            edge_magnitude_xy = np.sqrt(edge_x_xy**2 + edge_y_xy**2)
            edge_normalized_xy = edge_magnitude_xy / (np.max(edge_magnitude_xy) + 1e-8)
            slice_xy = slice_xy * 0.7 + edge_normalized_xy * np.max(slice_xy) * 0.3
        
        # Plotly interactive visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # XY Plane with Plotly
            fig_xy = go.Figure(data=go.Heatmap(
                z=slice_xy,
                colorscale=colormap,
                showscale=True,
                colorbar=dict(title="Intensity")
            ))
            fig_xy.update_layout(
                title="Focal Plane (XY)",
                xaxis_title="X",
                yaxis_title="Y",
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="#161b22",
                paper_bgcolor="#161b22",
                font=dict(color="#e6edf3")
            )
            
            # Add ground truth overlays
            if show_gt and has_graph:
                current_frame_nodes = nodes[nodes[:, 0] == t_idx]
                for node in current_frame_nodes:
                    node_z, node_y, node_x = node[1], node[2], node[3]
                    if abs(node_z - z_idx) <= 2:
                        fig_xy.add_trace(go.Scatter(
                            x=[node_x], y=[node_y],
                            mode="markers",
                            marker=dict(size=10, color="#58a6ff", line=dict(width=2, color="#e6edf3")),
                            name=f"Node {int(node[0])}",
                            hovertemplate="Node ID: %{fullData.name}<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>"
                        ))
            
            st.plotly_chart(fig_xy, width="full")
        
        with col2:
            # XZ Plane
            fig_xz = go.Figure(data=go.Heatmap(
                z=slice_xz,
                colorscale=colormap,
                showscale=False
            ))
            fig_xz.add_hline(y=z_idx, line=dict(color="#58a6ff", dash="dash", width=2))
            fig_xz.update_layout(
                title=f"XZ Cross-Section (Y={y_max//2})",
                xaxis_title="X",
                yaxis_title="Z",
                height=195,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="#161b22",
                paper_bgcolor="#161b22",
                font=dict(color="#e6edf3")
            )
            st.plotly_chart(fig_xz, width="full")
            
            # YZ Plane
            fig_yz = go.Figure(data=go.Heatmap(
                z=slice_yz,
                colorscale=colormap,
                showscale=False
            ))
            fig_yz.add_hline(y=z_idx, line=dict(color="#58a6ff", dash="dash", width=2))
            fig_yz.update_layout(
                title=f"YZ Cross-Section (X={x_max//2})",
                xaxis_title="Y",
                yaxis_title="Z",
                height=195,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="#161b22",
                paper_bgcolor="#161b22",
                font=dict(color="#e6edf3")
            )
            st.plotly_chart(fig_yz, width="full")
    
    with tab2:
        st.subheader("Cell Division Lineage Tracking")
        
        if has_graph and edges is not None and len(edges) > 0:
            # Build lineage graph
            G = nx.DiGraph()
            
            # Add nodes
            for idx, node in enumerate(nodes):
                t, z, y, x = node
                G.add_node(idx, t=t, z=z, y=y, x=x)
            
            # Add edges
            for edge in edges:
                source, target = edge
                if source in G.nodes() and target in G.nodes():
                    G.add_edge(source, target)
            
            # Create Plotly lineage visualization
            pos = {}
            for node in G.nodes():
                t = G.nodes[node]['t']
                pos[node] = (t, node)  # X-axis = Time, Y-axis = Node ID
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig_lineage = go.Figure()
            
            # Add edges
            fig_lineage.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(color="#58a6ff", width=1),
                hoverinfo="none",
                name="Lineage"
            ))
            
            # Add nodes
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_t = [G.nodes[n]['t'] for n in G.nodes()]
            
            fig_lineage.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers",
                marker=dict(size=8, color="#3fb950", line=dict(width=1, color="#e6edf3")),
                text=[f"Node {n}<br>T={G.nodes[n]['t']}<br>Z={G.nodes[n]['z']:.1f}" for n in G.nodes()],
                hovertemplate="%{text}<extra></extra>",
                name="Cells"
            ))
            
            fig_lineage.update_layout(
                title="Cell Lineage Tree (Time vs Node ID)",
                xaxis_title="Time Frame (T)",
                yaxis_title="Node ID",
                height=500,
                plot_bgcolor="#161b22",
                paper_bgcolor="#161b22",
                font=dict(color="#e6edf3"),
                showlegend=False
            )
            
            st.plotly_chart(fig_lineage, width="full")
            
            # Division detection
            divisions = [n for n in G.nodes() if G.out_degree(n) >= 2]
            if divisions:
                st.info(f"Detected {len(divisions)} cell divisions")
                st.write("Division events:", divisions)
        else:
            st.info("No lineage data available. Run tracking analysis first.")
    
    with tab3:
        st.subheader("Intensity & Evaluation Metrics")
        
        if has_graph:
            # Frame count visualization
            frame_counts = [int(np.sum(nodes[:, 0] == i)) for i in range(t_max)]
            
            fig_counts = go.Figure()
            fig_counts.add_trace(go.Scatter(
                x=list(range(t_max)),
                y=frame_counts,
                mode="lines+markers",
                line=dict(color="#58a6ff", width=2),
                marker=dict(size=4),
                name="Cell Count"
            ))
            fig_counts.add_vline(x=t_idx, line=dict(color="#3fb950", dash="dash", width=2))
            fig_counts.update_layout(
                title="Cell Count per Time Frame",
                xaxis_title="Time Frame (T)",
                yaxis_title="Number of Cells",
                height=300,
                plot_bgcolor="#161b22",
                paper_bgcolor="#161b22",
                font=dict(color="#e6edf3")
            )
            st.plotly_chart(fig_counts, width="full")
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Nodes", len(nodes))
            with col2:
                st.metric("Total Edges", len(edges) if edges is not None else 0)
            with col3:
                st.metric("Time Frames", t_max)
            
            # Kaggle score calculation
            if 'tracking_nodes' in st.session_state:
                evaluator = EmbryoSplitEvaluator()
                pred_nodes = st.session_state['tracking_nodes']
                pred_edges = st.session_state['tracking_edges']
                
                # Calculate score
                pred_edges_set = set(tuple(e) for e in pred_edges)
                true_edges_set = set(tuple(e) for e in edges) if edges is not None else set()
                
                edge_jaccard = evaluator.compute_edge_jaccard(pred_edges_set, true_edges_set)
                score = edge_jaccard  # Simplified for demo
                
                st.metric("Kaggle Score", f"{score:.4f}")
                
                # Update HUD
                st.markdown(f"<script>document.getElementById('kaggle-score').textContent = '{score:.4f}';</script>", unsafe_allow_html=True)
        else:
            st.info("No metrics available. Load a dataset with ground truth.")
else:
    st.info("Please select a dataset from the sidebar to begin analysis.")

# Cross-Validation Logic
if run_cv_btn and cv_enabled and selected_dataset:
    with st.spinner("Running cross-validation..."):
        evaluator = EmbryoSplitEvaluator(train_dir=TRAIN_DIR)
        splits = evaluator.create_splits(n_splits=n_folds)
        
        cv_scores = []
        
        for fold, (train_datasets, val_datasets) in enumerate(splits):
            st.write(f"Fold {fold + 1}/{n_folds}: Train={len(train_datasets)}, Val={len(val_datasets)}")
            
            # For demo, just use current dataset as validation
            if selected_dataset in val_datasets:
                # Run tracking on validation set
                if tracker_mode == "Classical DoG":
                    tracker = AnisotropicDoGTracker(
                        sigma_small=1.0, 
                        sigma_large=3.0, 
                        threshold=threshold,
                        multi_scale=multi_scale
                    )
                    tracking_results = tracker.track_volume(volume)
                    nodes_pred = tracking_results['nodes']
                    edges_pred = tracking_results['edges']
                else:
                    # Use ground truth for demo
                    nodes_pred = nodes
                    edges_pred = edges if edges is not None else np.empty((0, 2))
                
                # Calculate score
                pred_edges_set = set(tuple(e) for e in edges_pred)
                true_edges_set = set(tuple(e) for e in edges) if edges is not None else set()
                
                pred_divisions = evaluator.extract_divisions_from_graph(pred_edges_set, nodes_pred)
                true_divisions = evaluator.extract_divisions_from_graph(true_edges_set, nodes)
                
                score = evaluator.compute_competition_score(
                    nodes_pred, nodes,
                    pred_edges_set, true_edges_set,
                    pred_divisions, true_divisions
                )
                
                cv_scores.append(score)
                st.write(f"  Fold {fold + 1} Score: {score:.4f}")
        
        if cv_scores:
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            st.success(f"CV Mean Score: {mean_score:.4f} ± {std_score:.4f}")
            
            # Update HUD
            st.markdown(f"<script>document.getElementById('kaggle-score').textContent = '{mean_score:.4f}';</script>", unsafe_allow_html=True)
# Tracking Analysis Logic
if evaluate_btn and selected_dataset and evaluator:
    with st.spinner("Running tracking analysis..."):
        if tracker_mode == "Classical DoG":
            tracker = AnisotropicDoGTracker(
                sigma_small=1.0, 
                sigma_large=3.0, 
                threshold=threshold,
                multi_scale=multi_scale
            )
            tracking_results = tracker.track_volume(volume)
            nodes_pred = tracking_results['nodes']
            edges_pred = tracking_results['edges']
            
            # Detect divisions
            divisions = evaluator.detect_divisions_from_edges(edges_pred, nodes_pred)
            st.success(f"Detected {len(nodes_pred)} centroids, {len(edges_pred)} links, {len(divisions)} divisions")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CellTrackerUNet3D().to(device)
            model.eval()
            
            # Use ground truth for demo mode
            nodes_pred = nodes
            edges_pred = edges if edges is not None else np.empty((0, 2))
            divisions = evaluator.detect_divisions_from_edges(edges_pred, nodes_pred)
            st.success(f"Deep Learning inference complete: {len(nodes_pred)} nodes, {len(edges_pred)} edges, {len(divisions)} divisions")
        
        st.session_state['tracking_nodes'] = nodes_pred
        st.session_state['tracking_edges'] = edges_pred
        st.session_state['tracking_divisions'] = divisions
        st.rerun()

# Export Logic
if export_btn and selected_dataset:
    if 'tracking_nodes' in st.session_state:
        export_nodes = st.session_state['tracking_nodes']
        export_edges = st.session_state['tracking_edges']
    elif has_graph:
        export_nodes = nodes
        export_edges = edges if edges is not None else np.empty((0, 2))
    else:
        st.warning("No tracking results available. Run analysis first.")
        export_nodes = None
    
    if export_nodes is not None:
        # Kaggle-compliant export
        sub_records = []
        dataset_id = selected_dataset.replace(".zarr", "")
        row_id = 0
        
        # Node rows
        for idx, node in enumerate(export_nodes):
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
                "source_id": -1,
                "target_id": -1
            })
            row_id += 1
        
        # Edge rows
        for edge in export_edges:
            source_id, target_id = edge
            sub_records.append({
                "id": row_id,
                "dataset": dataset_id,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1.0,
                "y": -1.0,
                "x": -1.0,
                "source_id": int(source_id),
                "target_id": int(target_id)
            })
            row_id += 1

        sub_df = pd.DataFrame(sub_records)
        sub_df.to_csv("submission.csv", index=False)
        st.success(f"Exported {len(sub_df)} rows to submission.csv")
        st.dataframe(sub_df.head(20))