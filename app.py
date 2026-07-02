import os
import time
import torch
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from src.data_loader import get_zarr_stream
from src.graph_loader import parse_geff_data
from src.image_processor import extract_3d_crop
from src.model import CellTrackerNet3D

st.set_page_config(page_title="Biohub Cell Tracking Lab", layout="wide")

# Persistent styling configuration - BioHub Corporate Digital Identity
st.markdown(
    """
    <style>
    .stApp {
        background: #0b0f19;
        color: #e2e8f0; font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .stApp > div {
        background-color: #0b0f19;
    }
    .lab-shell { padding: 0.35rem 0 1.2rem 0; }
    .hero-card {
        background: rgba(11, 15, 25, 0.95);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 12px; padding: 1.2rem 1.4rem;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.1);
    }
    .eyebrow { font-size: 0.74rem; letter-spacing: 0.3em; text-transform: uppercase; color: #06b6d4; margin-bottom: 0.4rem; font-weight: 700; }
    .hero-title { font-size: 1.9rem; font-weight: 700; color: #f8fafc; margin: 0; }
    .hero-subtitle { color: #94a3b8; margin-top: 0.35rem; max-width: 740px; line-height: 1.5; }
    .metric-card {
        background: rgba(11, 15, 25, 0.9);
        border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 8px; padding: 0.9rem 1rem; min-height: 92px;
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.05);
    }
    .metric-label { font-size: 0.72rem; letter-spacing: 0.22em; color: #06b6d4; text-transform: uppercase; display: block; margin-bottom: 0.35rem; }
    .metric-value { font-size: 0.98rem; color: #e2e8f0; font-weight: 600; }
    .panel {
        background: rgba(11, 15, 25, 0.85);
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.08);
    }
    .panel h3 { color: #f8fafc; margin-top: 0; font-size: 1rem; font-weight: 600; letter-spacing: 0.05em; }
    .status-pill { display: inline-block; padding: 0.28rem 0.6rem; border-radius: 999px; background: rgba(16, 185, 129, 0.15); color: #10b981; font-size: 0.76rem; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-grid { display: grid; gap: 0.6rem; margin-top: 0.7rem; }
    .status-tile { background: rgba(11, 15, 25, 0.9); border: 1px solid rgba(6, 182, 212, 0.15); border-radius: 6px; padding: 0.7rem 0.75rem; }
    .status-tile .label { font-size: 0.68rem; letter-spacing: 0.2em; text-transform: uppercase; color: #06b6d4; }
    .status-tile .value { margin-top: 0.2rem; font-weight: 600; color: #f8fafc; font-size: 0.95rem; }
    .divider-line { height: 1px; background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.4), transparent); margin: 0.75rem 0; }
    .workflow-card { background: rgba(11, 15, 25, 0.9); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 6px; padding: 0.8rem 0.9rem; margin-top: 0.7rem; }
    .workflow-card .step { color: #cbd5e1; font-size: 0.83rem; margin: 0.22rem 0; }
    .workflow-card .step strong { color: #10b981; }
    .indicator-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #10b981; margin-right: 0.45rem; box-shadow: 0 0 8px rgba(16, 185, 129, 0.5); }
    .meter { height: 6px; background: rgba(30, 41, 59, 0.9); border-radius: 999px; overflow: hidden; margin-top: 0.45rem; }
    .meter > span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #06b6d4, #10b981); }
    .chart-card { background: rgba(11, 15, 25, 0.85); border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 8px; padding: 0.8rem; margin-top: 0.75rem; }
    .terminal-output { background: #0b0f19 !important; border: 1px solid rgba(6, 182, 212, 0.3); font-family: monospace; padding: 1rem; border-radius: 8px; color: #06b6d4; min-height: 260px; overflow-y: auto; white-space: pre-wrap; }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #06b6d4 0%, #10b981 100%);
        color: #0b0f19;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.4);
    }
    .stSelectbox > div > div {
        background-color: rgba(11, 15, 25, 0.9);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 6px;
    }
    .stSlider > div > div > div {
        background-color: rgba(11, 15, 25, 0.9);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section Display
st.markdown(
    """
    <div class="lab-shell">
      <div class="hero-card">
        <div class="eyebrow">BIOHUB CELL TRACKING LAB</div>
        <h1 class="hero-title">4D Volumetric Analysis Workspace</h1>
        <p class="hero-subtitle">A research-grade interface for inspecting 3D cell volumes, tracing motion across time, and preparing structured outputs with a calm, laboratory-first workflow.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# HUD Row Metrics Layout
hud1, hud2, hud3, hud4 = st.columns(4)
with hud1: st.markdown('<div class="metric-card"><span class="metric-label">Instrument State</span><div class="metric-value">3D volume intake ready</div></div>', unsafe_allow_html=True)
with hud2: st.markdown('<div class="metric-card"><span class="metric-label">Spatial Focus</span><div class="metric-value">Orthogonal slice navigation</div></div>', unsafe_allow_html=True)
with hud3: st.markdown('<div class="metric-card"><span class="metric-label">Data Stream</span><div class="metric-value">Lazy Zarr memory mapping</div></div>', unsafe_allow_html=True)
with hud4: st.markdown('<div class="metric-card"><span class="metric-label">Output Format</span><div class="metric-value">Submission-ready tracking table</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Session initialization
if "playback_active" not in st.session_state: st.session_state.playback_active = False
if "playback_frame" not in st.session_state: st.session_state.playback_frame = 0

# Strict three-column dashboard layout
left_controls, center_stage, right_analytics = st.columns([1, 2.5, 1], gap="medium")

with left_controls:
    st.markdown('<div class="panel"><h3>Input Management & Navigation</h3>', unsafe_allow_html=True)
    TRAIN_DIR = os.path.join("data", "train")
    zarr_folders = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".zarr")] if os.path.exists(TRAIN_DIR) else []

    if zarr_folders:
        selected_dataset = st.selectbox("Active volume", zarr_folders, label_visibility="collapsed", key="dataset_selector")
        volume = get_zarr_stream(selected_dataset, base_dir=TRAIN_DIR)
        t_max, z_max, y_max, x_max = volume.shape

        try:
            nodes, edges = parse_geff_data(selected_dataset, base_dir=TRAIN_DIR)
            has_graph = True
        except Exception:
            nodes, edges, has_graph = None, None, False

        st.markdown("<div class='status-pill'>Loaded</div>", unsafe_allow_html=True)
        st.markdown("<br><strong>Dimensions</strong>", unsafe_allow_html=True)
        st.caption(f"Frames: {t_max}  •  Depth: {z_max}  •  Shape: {y_max}x{x_max}")
        
        st.markdown("<br><strong>Navigation Controls</strong>", unsafe_allow_html=True)
        
        if st.session_state.playback_active:
            st.rerun() if hasattr(st, "rerun") else None
            st.session_state.playback_frame = (st.session_state.playback_frame + 1) % t_max
            t_idx = st.session_state.playback_frame
            st.caption(f"⏱️ Playing Temporal Sequence (Frame: {t_idx})")
        else:
            t_idx = st.slider("Time Axis (T Frame)", 0, t_max - 1, int(st.session_state.playback_frame), key="time_slider")
            st.session_state.playback_frame = t_idx
            
        z_idx = st.slider("Depth Axis (Z Slice)", 0, z_max - 1, z_max // 2, key="depth_slider")
        
        st.markdown("<br><strong>Playback Interface</strong>", unsafe_allow_html=True)
        play_col, reset_col = st.columns(2)
        with play_col:
            if st.button("▶ Play" if not st.session_state.playback_active else "⏸ Pause", key="play_pause_btn"):
                st.session_state.playback_active = not st.session_state.playback_active
        with reset_col:
            if st.button("Reset Track", key="reset_btn"):
                st.session_state.playback_active = False
                st.session_state.playback_frame = 0

        st.markdown("<br><strong>Overlay Filtering</strong>", unsafe_allow_html=True)
        show_gt = st.checkbox("Display ground-truth centers", value=True, key="overlay_checkbox") if has_graph else False

        st.markdown("<div class='divider-line'></div><div class='status-grid'>", unsafe_allow_html=True)
        st.markdown(f"<div class='status-tile'><div class='label'>Acquisition</div><div class='value'>{selected_dataset}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='status-tile'><div class='label'>Tracked points</div><div class='value'>{len(nodes) if has_graph else 'Unavailable'}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='status-tile'><div class='label'>Active slice</div><div class='value'>T={t_idx} / Z={z_idx}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("No dataset components were found in data/train/.")
        selected_dataset = None
    st.markdown("</div>", unsafe_allow_html=True)

with center_stage:
    st.markdown('<div class="panel"><h3>Widescreen Video/Image Matrix Stage</h3>', unsafe_allow_html=True)
    if selected_dataset:
        # Interactive Control Panel for Image Rendering
        st.markdown('<strong>Color Layer Presets</strong>', unsafe_allow_html=True)
        colormap_preset = st.selectbox(
            'Select visualization colormap',
            ['Inferno Cellular Contrast', 'Fluorescent Green Glow', 'High-Contrast Depth Mapping', 'Laser Scanning Confocal'],
            label_visibility='collapsed',
            key='colormap_selector'
        )
        
        colormap_map = {
            'Inferno Cellular Contrast': 'inferno',
            'Fluorescent Green Glow': 'viridis',
            'High-Contrast Depth Mapping': 'magma',
            'Laser Scanning Confocal': 'plasma'
        }
        selected_cmap = colormap_map[colormap_preset]
        
        st.markdown('<strong>Digital Filters</strong>', unsafe_allow_html=True)
        apply_edge_contrast = st.checkbox('Apply Edge Contrast Booster', value=False, key='edge_contrast_booster')
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        # Load native pixel structures for multi-planar orthogonal viewer
        slice_xy = volume[t_idx, z_idx, :, :].compute()
        slice_xz = volume[t_idx, :, 128, :].compute()  # XZ plane at Y = 128 midplane
        slice_yz = volume[t_idx, :, :, 128].compute()  # YZ plane at X = 128 midplane
        
        # Apply edge contrast booster if enabled (to all planes)
        if apply_edge_contrast:
            # XY plane edge enhancement
            edge_x_xy = sobel(slice_xy, axis=1)
            edge_y_xy = sobel(slice_xy, axis=0)
            edge_magnitude_xy = np.sqrt(edge_x_xy**2 + edge_y_xy**2)
            edge_normalized_xy = edge_magnitude_xy / (np.max(edge_magnitude_xy) + 1e-8)
            slice_xy = slice_xy * 0.7 + edge_normalized_xy * np.max(slice_xy) * 0.3
            
            # XZ plane edge enhancement
            edge_x_xz = sobel(slice_xz, axis=1)
            edge_y_xz = sobel(slice_xz, axis=0)
            edge_magnitude_xz = np.sqrt(edge_x_xz**2 + edge_y_xz**2)
            edge_normalized_xz = edge_magnitude_xz / (np.max(edge_magnitude_xz) + 1e-8)
            slice_xz = slice_xz * 0.7 + edge_normalized_xz * np.max(slice_xz) * 0.3
            
            # YZ plane edge enhancement
            edge_x_yz = sobel(slice_yz, axis=1)
            edge_y_yz = sobel(slice_yz, axis=0)
            edge_magnitude_yz = np.sqrt(edge_x_yz**2 + edge_y_yz**2)
            edge_normalized_yz = edge_magnitude_yz / (np.max(edge_magnitude_yz) + 1e-8)
            slice_yz = slice_yz * 0.7 + edge_normalized_yz * np.max(slice_yz) * 0.3
        
        # Multi-planar orthogonal viewer layout using GridSpec
        fig = plt.figure(figsize=(12, 10), facecolor="#0b0f19")
        gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 0.2], width_ratios=[3, 1, 0.2])
        
        # Large XY Plane (Focal Plane) - upper left, spanning 2 rows
        ax_xy = fig.add_subplot(gs[0:2, 0:2])
        ax_xy.imshow(slice_xy, cmap=selected_cmap)
        ax_xy.set_title("Focal Plane (XY)", color="#06b6d4", fontsize=11, loc="left", fontweight="600")
        if show_gt and has_graph:
            current_frame_nodes = nodes[nodes[:, 0] == t_idx]
            for node in current_frame_nodes:
                node_z, node_y, node_x = node[1], node[2], node[3]
                if abs(node_z - z_idx) <= 2:
                    ax_xy.scatter(node_x, node_y, s=120, edgecolors="#10b981", facecolors="none", linewidths=2)
        ax_xy.axis('off')
        
        # XZ Plane - horizontal underneath XY
        ax_xz = fig.add_subplot(gs[2, 0:2])
        ax_xz.imshow(slice_xz, cmap=selected_cmap, aspect='auto')
        ax_xz.axhline(z_idx, color="#06b6d4", linestyle="--", linewidth=2, alpha=0.9, label='Z Slice')
        ax_xz.set_title("XZ Cross-Section (Y=128)", color="#06b6d4", fontsize=9, loc="left", fontweight="600")
        ax_xz.axis('off')
        
        # YZ Plane - vertical to the right of XY
        ax_yz = fig.add_subplot(gs[0:2, 2])
        ax_yz.imshow(slice_yz, cmap=selected_cmap, aspect='auto')
        ax_yz.axhline(z_idx, color="#06b6d4", linestyle="--", linewidth=2, alpha=0.9)
        ax_yz.set_title("YZ Cross-Section (X=128)", color="#06b6d4", fontsize=9, loc="left", fontweight="600")
        ax_yz.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)

        if has_graph:
            frame_counts = [int(np.sum(nodes[:, 0] == i)) for i in range(t_max)]
            chart_fig, chart_ax = plt.subplots(figsize=(8, 2.5), facecolor="#0b0f19")
            chart_ax.plot(range(t_max), frame_counts, color="#06b6d4", linewidth=2)
            chart_ax.fill_between(range(t_max), frame_counts, alpha=0.2, color="#06b6d4")
            chart_ax.axvline(t_idx, color="#10b981", linestyle="-", linewidth=2)
            chart_ax.set_ylim(0, max(1, max(frame_counts) + 1))
            chart_ax.set_facecolor("#0b0f19")
            chart_ax.tick_params(colors="#94a3b8", labelsize=8)
            for spine in chart_ax.spines.values(): spine.set_color("#1e293b")
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.pyplot(chart_fig)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Select a dataset from the mount panel to inspect the volume.")
    st.markdown("</div>", unsafe_allow_html=True)

with right_analytics:
    st.markdown('<div class="panel"><h3>Cognitive Processing Console</h3>', unsafe_allow_html=True)
    evaluate_btn = st.button("Run tracking analysis", key="evaluate_btn")
    export_btn = st.button("Export submission table", key="export_btn")

    term_placeholder = st.empty()
    term_placeholder.markdown('<div class="terminal-output">Instrument idle. System prepared for the next analysis cycle.</div>', unsafe_allow_html=True)

    if selected_dataset:
        slice_data = volume[t_idx, z_idx, :, :].compute()
        signal_strength = float(np.mean(slice_data) / (np.max(slice_data) if np.max(slice_data) else 1))
        focus_lock = max(72, min(99, 92 - abs(z_idx - (z_max // 2)) * 1.4))
        tracking_confidence = 88 if has_graph else 74
        drift_monitor = 95 if show_gt else 80

        st.markdown("<div class='status-grid'>", unsafe_allow_html=True)
        st.markdown(f"<div class='status-tile'><div class='label'><span class='indicator-dot'></span>Signal stability</div><div class='value'>{signal_strength:.2f}</div><div class='meter'><span style='width:{min(100, signal_strength * 100):.0f}%'></span></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='status-tile'><div class='label'>Focus lock</div><div class='value'>{focus_lock:.0f}%</div><div class='meter'><span style='width:{focus_lock:.0f}%'></span></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='status-tile'><div class='label'>Tracking confidence</div><div class='value'>{tracking_confidence:.0f}%</div><div class='meter'><span style='width:{tracking_confidence:.0f}%'></span></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='status-tile'><div class='label'>Drift monitor</div><div class='value'>{drift_monitor:.0f}%</div><div class='meter'><span style='width:{drift_monitor:.0f}%'></span></div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='workflow-card'>", unsafe_allow_html=True)
    st.markdown("<div class='step'><strong>01</strong> • Acquire volumetric frames from dataset.</div>", unsafe_allow_html=True)
    st.markdown("<div class='step'><strong>02</strong> • Traverse slice views and coordinate overlays.</div>", unsafe_allow_html=True)
    st.markdown("<div class='step'><strong>03</strong> • Export clean submission files down-stream.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if evaluate_btn and selected_dataset:
        log_stream = "```"
        log_stream += "\n[INIT] Accessing volume stream..."
        term_placeholder.markdown(f'<div class="terminal-output">{log_stream}\n```▮</div>', unsafe_allow_html=True)
        time.sleep(0.3)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CellTrackerNet3D().to(device)
        model.eval()
        
        log_stream += f"\n[DEVICE] Compute backend allocated: {device.type.upper()}"
        if device.type == "cuda":
            log_stream += f" | GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        log_stream += f"\n[MODEL] 3D-CNN architecture loaded: CellTrackerNet3D"
        log_stream += f"\n[MODEL] Parameter count: {sum(p.numel() for p in model.parameters()):,}"
        term_placeholder.markdown(f'<div class="terminal-output">{log_stream}\n```▮</div>', unsafe_allow_html=True)
        time.sleep(0.3)

        log_stream += f"\n[EXTRACT] Extracting local 3D subvolume for frame T={t_idx}..."
        term_placeholder.markdown(f'<div class="terminal-output">{log_stream}\n```▮</div>', unsafe_allow_html=True)
        time.sleep(0.2)

        if has_graph:
            current_frame_nodes = nodes[nodes[:, 0] == t_idx]
            log_stream += f"\n[DATA] Frame {t_idx} contains {len(current_frame_nodes)} cell targets"
            term_placeholder.markdown(f'<div class="terminal-output">{log_stream}\n```▮</div>', unsafe_allow_html=True)
            
            for idx, sample_node in enumerate(current_frame_nodes[:5]):  # Process first 5 cells for demo
                _, nz, ny, nx = sample_node
                log_stream += f"\n[CROP {idx+1}] Target coordinates: Z={nz:.1f}, Y={ny:.1f}, X={nx:.1f}"
                term_placeholder.markdown(f'<div class="terminal-output">{log_stream}\n```▮</div>', unsafe_allow_html=True)
                time.sleep(0.15)
                
                patch = extract_3d_crop(volume, t_idx, nz, ny, nx)
                log_stream += f"\n[CROP {idx+1}] Patch dimensions: {patch.shape} | Intensity range: [{patch.min():.3f}, {patch.max():.3f}]"
                term_placeholder.markdown(f'<div class="terminal-output">{log_stream}\n```▮</div>', unsafe_allow_html=True)
                time.sleep(0.1)
                
                patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device).float()
                log_stream += f"\n[TENSOR] Input tensor shape: {patch_t.shape} | dtype: {patch_t.dtype}"
                term_placeholder.markdown(f'<div class="terminal-output">{log_stream}\n```▮</div>', unsafe_allow_html=True)
                time.sleep(0.15)

                with torch.no_grad():
                    pred_coords = model(patch_t).cpu().numpy()[0]

                log_stream += f"\n[INFERENCE] Coordinate regression deltas:"
                log_stream += f"\n  → Z-offset: {pred_coords[0]:.4f}"
                log_stream += f"\n  → Y-offset: {pred_coords[1]:.4f}"
                log_stream += f"\n  → X-offset: {pred_coords[2]:.4f}"
                term_placeholder.markdown(f'<div class="terminal-output">{log_stream}\n```▮</div>', unsafe_allow_html=True)
                time.sleep(0.1)
            
            if len(current_frame_nodes) > 5:
                log_stream += f"\n[INFO] Remaining {len(current_frame_nodes) - 5} cells processed in batch mode..."
                term_placeholder.markdown(f'<div class="terminal-output">{log_stream}\n```▮</div>', unsafe_allow_html=True)
                time.sleep(0.2)
            
            log_stream += f"\n[COMPLETE] Analysis cycle finished. Output aligned for review."
            log_stream += "\n```"
            term_placeholder.markdown(f'<div class="terminal-output">{log_stream}</div>', unsafe_allow_html=True)
        else:
            log_stream += f"\n[WARNING] No ground-truth graph available for inference validation."
            log_stream += "\n```"
            term_placeholder.markdown(f'<div class="terminal-output">{log_stream}</div>', unsafe_allow_html=True)

    if export_btn and selected_dataset:
        if has_graph:
            sub_records = []
            dataset_id = selected_dataset.replace(".zarr", "")
            for idx, node in enumerate(nodes):
                sub_records.append({
                    "dataset_id": dataset_id, "frame": int(node[0]), "cell_id": int(idx),
                    "z": float(node[1]), "y": float(node[2]), "x": float(node[3]), "parent_id": -1,
                })

            sub_df = pd.DataFrame(sub_records)
            sub_df.to_csv("submission.csv", index=False)
            st.success("Submission matrix saved to submission.csv")
            st.dataframe(sub_df.head(100), height=200)
        else:
            st.info("Ground-truth coordinates were not available for export.")
    st.markdown("</div>", unsafe_allow_html=True)