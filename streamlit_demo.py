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
                    auto-rotate
                    auto-rotate-delay="0"
                    rotation-per-second="-30deg"
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


if __name__ == '__main__':
    main()
