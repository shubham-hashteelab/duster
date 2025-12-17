#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dust3r streamlit demo
# --------------------------------------------------------
import os
import torch
import tempfile
import streamlit as st
import numpy as np
import copy
import base64
from pathlib import Path

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.demo import _convert_scene_output_to_glb

import matplotlib.pyplot as pl
import pandas as pd
from scipy.spatial import cKDTree
import trimesh

torch.backends.cuda.matmul.allow_tf32 = True


@st.cache_resource
def load_model(model_name, device):
    """Load and cache the model"""
    weights_path = "naver/" + model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)
    return model


def get_3D_model_from_scene(outdir, scene, min_conf_thr=3, as_pointcloud=False, 
                           mask_sky=False, clean_depth=False, transparent_cams=False, cam_size=0.05):
    """Extract 3D model (glb file) from a reconstructed scene"""
    if scene is None:
        return None, None, None
    
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    
    outfile = _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, 
                                          as_pointcloud=as_pointcloud, transparent_cams=transparent_cams, 
                                          cam_size=cam_size, silent=True)
    
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return outfile, imgs, scene


def calculate_distance_profile(glb_file, point_coords, slice_thickness=0.12, n_bins=80, min_points_per_bin=5):
    """Calculate roof-to-floor distance profile from GLB file and 4 picked points"""
    
    # Load GLB as point cloud
    mesh = trimesh.load(glb_file)
    if hasattr(mesh, 'vertices'):
        points = np.array(mesh.vertices)
    else:
        # If it's a scene with multiple meshes, concatenate all vertices
        points = np.vstack([m.vertices for m in mesh.geometry.values()])
    
    # Extract 4 points: roof1, roof2, floor1, floor2
    roof1, roof2 = np.array(point_coords[0]), np.array(point_coords[1])
    floor1, floor2 = np.array(point_coords[2]), np.array(point_coords[3])
    
    # Extract slices
    def extract_slice(p1, p2, thickness=slice_thickness):
        dir_vec = p2 - p1
        L = np.linalg.norm(dir_vec)
        if L < 0.01:
            return None, None, None
        dir_vec /= L
        vecs = points - p1
        proj = np.dot(vecs, dir_vec)
        closest = p1 + proj[:, None] * dir_vec
        dist = np.linalg.norm(points - closest, axis=1)
        mask = dist <= thickness
        t = np.clip(proj[mask] / L, 0.0, 1.0)
        return points[mask], t, L
    
    roof_pts, roof_t, L_roof = extract_slice(roof1, roof2)
    floor_pts, floor_t, L_floor = extract_slice(floor1, floor2)
    
    if roof_pts is None or floor_pts is None or len(roof_pts) < 10 or len(floor_pts) < 10:
        return None, None
    
    ref_length = (L_roof + L_floor) / 2
    
    # Robust pairing + de-trend
    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    
    dist_along = []
    raw_distances = []
    
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        roof_in_bin = roof_pts[(roof_t >= lo) & (roof_t < hi)]
        floor_in_bin = floor_pts[(floor_t >= lo) & (floor_t < hi)]
        
        if len(roof_in_bin) < min_points_per_bin or len(floor_in_bin) < min_points_per_bin:
            continue
        
        roof_median = np.median(roof_in_bin, axis=0)
        floor_median = np.median(floor_in_bin, axis=0)
        distance = np.linalg.norm(roof_median - floor_median)
        
        dist_along.append(centers[i] * ref_length)
        raw_distances.append(distance)
    
    dist_along = np.array(dist_along)
    raw_distances = np.array(raw_distances)
    
    if len(raw_distances) < 10:
        return None, None
    
    # Remove false linear trend
    p = np.polyfit(dist_along, raw_distances, 1)
    trend = np.polyval(p, dist_along)
    detrended = raw_distances - trend + np.mean(raw_distances)
    
    # Adaptive smoothing
    smooth = pd.Series(detrended).rolling(window=7, center=True, min_periods=1).median()
    smooth = smooth.rolling(window=9, center=True, min_periods=1).mean()
    
    # Stats
    mean_h = np.mean(detrended)
    std_h = np.std(detrended)
    min_h = np.min(detrended)
    max_h = np.max(detrended)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(dist_along, detrended, 'o', color='lightsteelblue', ms=6, alpha=0.7, label="Detrended measurements")
    ax.plot(dist_along, smooth, '-', color='navy', linewidth=4, label=f"Smoothed (±{std_h:.3f} m)")
    
    ax.axhspan(mean_h - 0.05, mean_h + 0.05, color='green', alpha=0.15, label="±5 cm (excellent)")
    ax.axhspan(mean_h - 0.10, mean_h + 0.10, color='yellow', alpha=0.10)
    ax.axhline(mean_h, color='black', linestyle='--', linewidth=1.5, label=f"Mean = {mean_h:.3f} m")
    
    ax.set_title(f"Roof-to-Floor Clearance Profile (trend removed)\n"
              f"Length ≈ {ref_length:.2f} m  |  Mean height = {mean_h:.3f} m  |  "
              f"Min {min_h:.3f} m  |  Max {max_h:.3f} m  |  σ = {std_h:.3f} m",
              fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("Distance Along Tunnel (m)", fontsize=13)
    ax.set_ylabel("Clearance Height (m)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(mean_h - 0.25, mean_h + 0.25)
    
    # Create dataframe
    df = pd.DataFrame({
        "distance_along_m": dist_along,
        "height_raw_m": raw_distances,
        "height_detrended_m": detrended,
        "height_smooth_m": smooth
    })
    
    return fig, df


def reconstruct_scene(model, device, image_size, uploaded_files, tmpdir,
                     schedule='linear', niter=300, scenegraph_type='complete',
                     min_conf_thr=3.0, as_pointcloud=False, mask_sky=False,
                     clean_depth=True, transparent_cams=False, cam_size=0.05):
    """Run dust3r inference and global alignment"""
    
    # Save uploaded files temporarily
    filelist = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(tmpdir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        filelist.append(file_path)
    
    try:
        square_ok = model.square_ok
    except:
        square_ok = False
    
    imgs = load_images(filelist, size=image_size, verbose=False, 
                      patch_size=model.patch_size, square_ok=square_ok)
    
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    
    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=False)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=0.01)

    return get_3D_model_from_scene(tmpdir, scene, min_conf_thr, as_pointcloud, 
                                  mask_sky, clean_depth, transparent_cams, cam_size)


def main():
    st.set_page_config(page_title="HashteeLab 3D Demo", layout="wide")
    st.title("HashteeLab 3D Demo")
    
    # Initialize session state
    if 'scene' not in st.session_state:
        st.session_state.scene = None
    if 'tmpdir' not in st.session_state:
        st.session_state.tmpdir = tempfile.mkdtemp(suffix='dust3r_streamlit_demo')
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("Configuration")
        device = st.selectbox("Device", ["cuda", "cpu"], index=0)
        image_size = st.selectbox("Image Size", [512, 224], index=0)
    
    # Set default model
    model_name = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_name, device)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    # Processing options
    col1, col2 = st.columns(2)
    with col1:
        as_pointcloud = st.checkbox("Export as Point Cloud", value=True)
        mask_sky = st.checkbox("Remove Sky from Scene", value=True)
    with col2:
        clean_depth = st.checkbox("Apply Depth Map Cleanup", value=True)
        transparent_cams = st.checkbox("Show Cameras with Transparency", value=True)
    
    # Confidence threshold slider
    min_conf_thr = st.slider(
        "Minimum Confidence Threshold", 
        min_value=1.0, 
        max_value=10.0, 
        value=3.0, 
        step=0.5,
        help="Only show points with confidence above this threshold. Higher values show fewer but more confident points."
    )
    
    # Run button
    if st.button("Run 3D Reconstruction", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one image")
        else:
            with st.spinner("Processing images..."):
                try:
                    # Set cam_size to 0 when showing as pointcloud to hide camera frustums
                    cam_size_value = 0.0 if as_pointcloud else 0.05
                    outfile, imgs, scene = reconstruct_scene(
                        model, device, image_size, uploaded_files, st.session_state.tmpdir,
                        min_conf_thr=min_conf_thr, as_pointcloud=as_pointcloud,
                        mask_sky=mask_sky, clean_depth=clean_depth,
                        transparent_cams=transparent_cams, cam_size=cam_size_value
                    )
                    st.session_state.scene = scene
                    st.session_state.outfile = outfile
                    st.session_state.imgs = imgs
                    st.success("Reconstruction complete!")
                except Exception as e:
                    st.error(f"Error during reconstruction: {str(e)}")
    
    # Display results
    if st.session_state.scene is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Images: RGB, Depth, Confidence")
            if 'imgs' in st.session_state:
                for i in range(0, len(st.session_state.imgs), 3):
                    subcol1, subcol2, subcol3 = st.columns(3)
                    with subcol1:
                        st.image(st.session_state.imgs[i], caption="RGB", use_container_width=True)
                    if i+1 < len(st.session_state.imgs):
                        with subcol2:
                            st.image(st.session_state.imgs[i+1], caption="Depth", use_container_width=True)
                    if i+2 < len(st.session_state.imgs):
                        with subcol3:
                            st.image(st.session_state.imgs[i+2], caption="Confidence", use_container_width=True)
        
        with col2:
            st.subheader("3D Reconstruction")
            if 'outfile' in st.session_state and os.path.exists(st.session_state.outfile):
                # Read GLB file and encode as base64
                with open(st.session_state.outfile, "rb") as f:
                    glb_data = f.read()
                    glb_base64 = base64.b64encode(glb_data).decode()
                
                # Embed 3D viewer using model-viewer
                viewer_html = f"""
                <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
                <model-viewer 
                    src="data:model/gltf-binary;base64,{glb_base64}"
                    alt="3D Model"
                    camera-controls
                    camera-orbit="0deg 75deg 105%"
                    style="width: 100%; height: 600px; background-color: #f0f0f0;"
                    shadow-intensity="1">
                </model-viewer>
                """
                st.components.v1.html(viewer_html, height=620)
                
                # Download button below viewer
                st.download_button(
                    label="Download 3D Model (GLB)",
                    data=glb_data,
                    file_name="scene.glb",
                    mime="model/gltf-binary"
                )
        
        # Distance Analysis Section
        st.divider()
        st.subheader("🔍 Distance Analysis (Roof-to-Floor/Wall-to-Wall)")
        st.info("Enter 4 points as [x, y, z] coordinates: Roof Start, Roof End, Floor Start, Floor End")
        
        with st.expander("ℹ️ How to get coordinates", expanded=False):
            st.markdown("""
            **To find point coordinates:**
            1. Download the GLB file above
            2. Open it in a 3D viewer (e.g., Blender, MeshLab, or online viewers like glTF Viewer)
            3. Select points and note their [x, y, z] coordinates
            4. Enter the coordinates below in the format: x, y, z (separated by commas)
            
            **Point Order:**
            - Point 1: Roof start
            - Point 2: Roof end  
            - Point 3: Floor start
            - Point 4: Floor end
            """)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p1_input = st.text_input("Point 1 (Roof Start)", placeholder="x, y, z", key="p1")
            p3_input = st.text_input("Point 3 (Floor Start)", placeholder="x, y, z", key="p3")
        with col_p2:
            p2_input = st.text_input("Point 2 (Roof End)", placeholder="x, y, z", key="p2")
            p4_input = st.text_input("Point 4 (Floor End)", placeholder="x, y, z", key="p4")
        
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            slice_thickness = st.slider("Slice Thickness (m)", 0.05, 0.50, 0.12, 0.01)
        with col_param2:
            n_bins = st.slider("Number of Bins", 20, 150, 80, 10)
        
        if st.button("Calculate Distance Profile", type="primary"):
            try:
                # Parse coordinates
                points = []
                for p_input in [p1_input, p2_input, p3_input, p4_input]:
                    coords = [float(x.strip()) for x in p_input.split(',')]
                    if len(coords) != 3:
                        raise ValueError("Each point must have exactly 3 coordinates")
                    points.append(coords)
                
                with st.spinner("Calculating distance profile..."):
                    fig, df = calculate_distance_profile(
                        st.session_state.outfile, 
                        points, 
                        slice_thickness=slice_thickness,
                        n_bins=n_bins
                    )
                    
                    if fig is None:
                        st.error("Not enough overlapping sections. Try increasing slice thickness or adjusting points.")
                    else:
                        st.session_state.distance_fig = fig
                        st.session_state.distance_df = df
                        st.success("Distance profile calculated!")
                        
            except ValueError as e:
                st.error(f"Invalid input format: {str(e)}")
            except Exception as e:
                st.error(f"Error calculating distance profile: {str(e)}")
        
        # Display results
        if 'distance_fig' in st.session_state:
            st.pyplot(st.session_state.distance_fig)
            
            if 'distance_df' in st.session_state:
                st.subheader("📊 Data Table")
                st.dataframe(st.session_state.distance_df, use_container_width=True)
                
                # Download CSV
                csv = st.session_state.distance_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Data",
                    data=csv,
                    file_name="distance_profile.csv",
                    mime="text/csv"
                )


if __name__ == '__main__':
    main()
