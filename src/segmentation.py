# -------------------------------------------------------------------------------
# SegmentThat: Interactive Gaussian Splat Segmentation
#
# Copyright (C) 2026 Filmakademie Baden-Württemberg,
# Carl Laemmle Institut for Research and Media Innovation
# https://github.com/FilmakademieRnd/SegmentThat_InteractiveTool
#
# SegmentThat is developed by the Carl Laemmle Institut for Research and Media
# Innovation in the context of the Industrial Metaverse Reallabor project
# (https://cli.filmakademie.de/imrl), funded by the Ministry of Economic Affairs,
# Skilled Crafts and Tourism Baden-Württemberg.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the MIT License for more details.
# You should have received a copy of the MIT License along with this program;
# if not go to https://opensource.org/licenses/MIT
# -------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
#
# Author: Sevinc Eroglu
# Date: 26.10.2026


# Standard library
import traceback
from pathlib import Path

# Third-party
import gsplat
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, maximum_flow
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Local
import state
from ply_io import read_ply, write_ply
from state import (
    CUT_BACKGROUND_PRIOR,
    CUT_BAND_FRACTION,
    CUT_BAND_RADIUS,
    CUT_EVIDENCE_SATURATION,
    CUT_FLOW_SCALE,
    CUT_MAX_EXTENT_MULTIPLIER,
    CUT_PAIRWISE_COLOR_SIGMA,
    CUT_PAIRWISE_K,
    CUT_PAIRWISE_RADIUS,
    CUT_PAIRWISE_SPATIAL_SIGMA,
    CUT_PAIRWISE_WEIGHT,
    CUT_UNARY_WEIGHT,
    DEVICE,
    MAX_CUT_CANDIDATES,
    PENDING_HIGHLIGHT_COLOR,
    VOTE_ANCHOR_THRESHOLD,
)

# =============================================================================
# 1. LOGGING
# =============================================================================


def log(x):
    """Print a prefixed log message."""
    print(f"[SegmentThat] {x}", flush=True)


def log_exception(x):
    log(x)
    traceback.print_exc()


# =============================================================================
# 2. SCENE LOADING AND CAMERA MATHS
# =============================================================================


def load_splat(file_path):
    if file_path is None:
        return None, "No file selected."
    path = Path(file_path)
    if not path.exists():
        return None, f"File does not exist: {path}"
    if path.suffix.lower() not in {".ply", ".splat"}:
        return None, "Please select a .ply or .splat file."
    return str(path), f"Loaded: {path.name}\nSize: {path.stat().st_size/(1024**2):.1f} MB"


def load_ply(ply_path):
    """Read a 3DGS .ply and return (means, quats, scales, opacities, colors) tensors on DEVICE."""
    log(f"Reading PLY: {ply_path}")
    v = read_ply(ply_path).vertices
    means_np = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(
        np.float32
    )
    quats_np = np.stack(
        [
            np.asarray(v["rot_0"]),
            np.asarray(v["rot_1"]),
            np.asarray(v["rot_2"]),
            np.asarray(v["rot_3"]),
        ],
        1,
    ).astype(np.float32)
    scales_np = np.exp(
        np.stack(
            [np.asarray(v["scale_0"]), np.asarray(v["scale_1"]), np.asarray(v["scale_2"])], 1
        ).astype(np.float32)
    )
    opacity_np = np.asarray(v["opacity"]).astype(np.float32)
    opacity_np = 1.0 / (1.0 + np.exp(-opacity_np))
    c0 = 0.28209479177387814
    f_dc = np.stack(
        [np.asarray(v["f_dc_0"]), np.asarray(v["f_dc_1"]), np.asarray(v["f_dc_2"])], 1
    ).astype(np.float32)
    colors_np = np.clip(0.5 + c0 * f_dc, 0.0, 1.0)
    return tuple(
        torch.from_numpy(x).to(DEVICE)
        for x in [means_np, quats_np, scales_np, opacity_np, colors_np]
    )


def quats_scales_to_covariances(quats_np, scales_np):
    """Build per-Gaussian 3x3 covariance matrices from quaternions (wxyz) and scales."""
    rotmats = Rotation.from_quat(quats_np[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)
    scaled = rotmats * scales_np[:, None, :]
    return np.einsum("nij,nkj->nik", scaled, scaled).astype(np.float32)


def viewmat_from_camera_pose(position, wxyz):
    """Convert a Viser camera pose (camera-to-world) to a 4x4 world-to-camera matrix."""
    w, x, y, z = [float(v) for v in wxyz]
    r_c2w = Rotation.from_quat([x, y, z, w]).as_matrix().astype(np.float32)
    pos = np.asarray(position, dtype=np.float32)
    r_w2c = r_c2w.T
    t_w2c = -r_w2c @ pos
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = r_w2c
    T[:3, 3] = t_w2c
    return torch.from_numpy(T).to(DEVICE)


# =============================================================================
# 3. RENDERING AND PREVIEW
# =============================================================================


def effective_opacities(o):
    return o if state.clip_mask is None else o * state.clip_mask


def render_view(
    viewmat, colors_override=None, width=None, height=None, k=None, opacities_override=None
):
    """Render the scene with gsplat for the given world-to-camera matrix."""
    if state.means is None:
        raise RuntimeError("Scene has not been loaded.")
    c = state.colors if colors_override is None else colors_override
    o = effective_opacities(state.opacities if opacities_override is None else opacities_override)
    w = state.WIDTH if width is None else width
    h = state.HEIGHT if height is None else height
    kk = state.K if k is None else k
    with torch.inference_mode():
        rc, _, _ = gsplat.rasterization(
            means=state.means,
            quats=state.quats,
            scales=state.scales,
            opacities=o,
            colors=c,
            viewmats=viewmat[None],
            Ks=kk[None],
            width=w,
            height=h,
            near_plane=0.01,
            far_plane=100.0,
            render_mode="RGB",
            camera_model="pinhole",
        )
    return Image.fromarray(rc[0].clamp(0, 1).mul(255).byte().cpu().numpy())


def warmup_gsplat():
    """Render a one-Gaussian dummy scene so gsplat builds/loads its CUDA kernels at startup
    instead of on the first Capture View (the first build can take several minutes)."""
    log("Warming up gsplat (first run compiles CUDA kernels, this may take a few minutes)...")
    means = torch.tensor([[0.0, 0.0, 2.0]], device=DEVICE, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE)
    scales = torch.full((1, 3), 0.1, device=DEVICE)
    opacities = torch.ones(1, device=DEVICE)
    colors = torch.ones(1, 3, device=DEVICE)
    viewmat = torch.eye(4, device=DEVICE)
    k = torch.tensor([[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]], device=DEVICE)
    rc, _, _ = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None],
        Ks=k[None],
        width=16,
        height=16,
        render_mode="RGB",
        camera_model="pinhole",
    )
    # Also run the backward pass, which Segment uses for foreground attribution.
    rc.sum().backward()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    log("gsplat ready.")


def build_display_colors(pending_indices=None):
    if state.colors is None:
        return None
    vis = state.colors.clone()
    if state.object_records:
        palette = plt.cm.hsv(np.linspace(0, 0.92, max(len(state.object_records), 1)))
        for i, r in enumerate(state.object_records):
            idx = torch.from_numpy(r["gaussian_indices"]).to(DEVICE)
            vis[idx] = torch.tensor(palette[i, :3], device=DEVICE, dtype=state.colors.dtype)
    if pending_indices is not None and len(pending_indices):
        idx = torch.from_numpy(pending_indices).to(DEVICE)
        vis[idx] = torch.tensor(PENDING_HIGHLIGHT_COLOR, device=DEVICE, dtype=state.colors.dtype)
    return vis


def build_isolated_opacities(pending_indices):
    if state.opacities is None:
        return None
    vis = torch.zeros_like(state.opacities)
    if pending_indices is not None and len(pending_indices):
        idx = torch.from_numpy(pending_indices).to(DEVICE)
        vis[idx] = state.opacities[idx]
    return vis


def build_inverted_opacities(pending_indices):
    if state.opacities is None:
        return None
    vis = state.opacities.clone()
    if pending_indices is not None and len(pending_indices):
        idx = torch.from_numpy(pending_indices).to(DEVICE)
        vis[idx] = 0
    return vis


def render_preview(pending, isolate):
    if isolate == "isolate":
        c = state.colors
        o = build_isolated_opacities(pending)
    elif isolate == "invert":
        c = state.colors
        o = build_inverted_opacities(pending)
    else:
        c = build_display_colors(pending)
        o = None
    return render_view(
        state.preview_viewmat,
        c,
        state.preview_width,
        state.preview_height,
        state.preview_k,
        opacities_override=o,
    )


# =============================================================================
# 4. SPATIAL CONNECTED COMPONENTS
# =============================================================================


def spatial_connected_components(xyz, radius):
    """Group points connected within `radius` of each other (sparse graph over a KD-tree).

    Components are returned ordered by their smallest index, each with ascending indices.
    """
    n = len(xyz)
    if n == 0:
        return []
    if n == 1:
        return [np.array([0], dtype=np.int64)]
    pairs = cKDTree(xyz).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    return np.split(order.astype(np.int64), np.flatnonzero(np.diff(labels[order])) + 1)


# =============================================================================
# 5. MULTI-VIEW VOTING AND GRAPH CUT
# =============================================================================


def accumulate_view_weights(viewmat, mask, width, height, fx, fy, cx, cy, numer, denom):
    """Project Gaussians into a view and accumulate 2D mask votes into `numer` / `denom`."""
    k_view = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=DEVICE, dtype=torch.float32
    )
    signal = torch.ones(state.N, 1, device=DEVICE, dtype=torch.float32, requires_grad=True)
    rc, _, _ = gsplat.rasterization(
        means=state.means,
        quats=state.quats,
        scales=state.scales,
        opacities=effective_opacities(state.opacities),
        colors=signal,
        viewmats=viewmat[None],
        Ks=k_view[None],
        width=width,
        height=height,
        near_plane=0.01,
        far_plane=100.0,
        render_mode="RGB",
        camera_model="pinhole",
    )
    rc0 = rc[0, :, :, 0]
    mask_t = torch.from_numpy(mask).to(device=DEVICE, dtype=torch.float32)
    (grad_denom,) = torch.autograd.grad(rc0.sum(), signal, retain_graph=True)
    (grad_numer,) = torch.autograd.grad((rc0 * mask_t).sum(), signal)
    with torch.no_grad():
        denom += grad_denom.squeeze(1)
        numer += grad_numer.squeeze(1)


def accumulate_all_pending_views():
    numer = torch.zeros(state.N, dtype=torch.float32, device=DEVICE)
    denom = torch.zeros(state.N, dtype=torch.float32, device=DEVICE)
    for v in state.pending_views:
        accumulate_view_weights(
            v["viewmat"],
            v["mask"],
            v["width"],
            v["height"],
            v["fx"],
            v["fy"],
            v["cx"],
            v["cy"],
            numer,
            denom,
        )
    return numer.cpu().numpy(), denom.cpu().numpy()


def drop_isolated(indices, k, min_selected):
    """Split `indices` into Gaussians kept and those with < `min_selected` of their `k`
    nearest scene neighbours also selected (isolated strays)."""
    if not len(indices):
        return indices, indices
    selected = np.zeros(state.N, dtype=bool)
    selected[indices] = True
    _, nn = state.full_tree.query(state.means_cpu[indices], k=k + 1, workers=-1)
    keep = selected[nn[:, 1:]].sum(1) >= min_selected
    return indices[keep], indices[~keep]


def min_cut(source, sink, pairs, w):
    """Min-cut with SciPy's maximum_flow; returns True for nodes on the sink (background) side.

    SciPy needs integer capacities, so all weights are scaled by CUT_FLOW_SCALE and rounded.
    """
    n = len(source)
    s, t = n, n + 1
    idx = np.arange(n)
    rows = np.concatenate([np.full(n, s), idx, pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([idx, np.full(n, t), pairs[:, 1], pairs[:, 0]])
    cap = np.rint(np.concatenate([source, sink, w, w]) * CUT_FLOW_SCALE).astype(np.int32)
    keep = cap > 0
    graph = csr_matrix((cap[keep], (rows[keep], cols[keep])), shape=(n + 2, n + 2))
    flow = maximum_flow(graph, s, t, method="dinic").flow
    # Nodes still reachable from the source through unsaturated edges form the object side.
    residual = (graph - flow).tocsr()
    residual.data[residual.data < 0] = 0
    residual.eliminate_zeros()
    reached = breadth_first_order(residual, s, directed=True, return_predecessors=False)
    source_side = np.zeros(n + 2, dtype=bool)
    source_side[reached] = True
    return ~source_side[:n]


def graph_cut_object(seed_indices, numer, denom):
    """Refine seed Gaussians into a 3D object with a min-cut over multi-view votes.

    The data term of each Gaussian is scaled by how clearly it was seen in the masked
    views, so Gaussians that were never visible are labelled by their similar neighbours
    instead of being forced to background.
    """
    seed_indices = seed_indices[~state.claimed[seed_indices]]
    if seed_indices.size == 0:
        return seed_indices
    has_evidence = denom > 1e-6
    conf_all = np.zeros(state.N, np.float32)
    conf_all[has_evidence] = np.clip(numer[has_evidence] / denom[has_evidence], 0, 1)
    seed_xyz = state.means_cpu[seed_indices]
    seed_centroid = seed_xyz.mean(0)
    seed_extent = (
        float(np.linalg.norm(seed_xyz - seed_centroid, axis=1).max()) if len(seed_xyz) > 1 else 0.0
    )
    max_reach = seed_extent * CUT_MAX_EXTENT_MULTIPLIER + CUT_BAND_RADIUS
    voted = np.flatnonzero(has_evidence & (conf_all > VOTE_ANCHOR_THRESHOLD))
    voted = voted[~state.claimed[voted]]
    if voted.size:
        voted_dist = np.linalg.norm(state.means_cpu[voted] - seed_centroid, axis=1)
        voted = voted[voted_dist <= max_reach]
    anchor = np.union1d(seed_indices, voted) if voted.size else seed_indices
    xyz = state.means_cpu[anchor]
    centroid = xyz.mean(0)
    extent = float(np.linalg.norm(xyz - centroid, axis=1).max()) if len(xyz) > 1 else 0.0
    nearby = state.full_tree.query_ball_point(
        centroid, min(extent * (1.0 + CUT_BAND_FRACTION) + CUT_BAND_RADIUS, max_reach)
    )
    candidates = np.unique(np.concatenate([anchor, np.asarray(nearby, dtype=np.int64)]))
    candidates = candidates[~state.claimed[candidates]]
    if not candidates.size:
        return candidates
    if candidates.size > MAX_CUT_CANDIDATES:
        is_anchor = np.isin(candidates, anchor)
        s = candidates[is_anchor]
        ns = candidates[~is_anchor]
        budget = max(MAX_CUT_CANDIDATES - len(s), 0)
        if len(ns) > budget:
            # Keep the Gaussians closest/nearest to the anchors tp avoid holes in the shell.
            dist, _ = cKDTree(state.means_cpu[s]).query(state.means_cpu[ns], k=1, workers=-1)
            ns = ns[np.argsort(dist, kind="stable")[:budget]]
        candidates = np.union1d(s, ns)
    conf = conf_all[candidates]
    # Evidence strength relative to the seeds: 1 for clearly seen Gaussians, 0 for unseen ones.
    seen_ref = float(np.median(denom[seed_indices])) * CUT_EVIDENCE_SATURATION
    evidence = np.clip(denom[candidates] / max(seen_ref, 1e-6), 0.0, 1.0)
    source = (CUT_UNARY_WEIGHT * evidence * conf).astype(np.float64)
    sink = (CUT_UNARY_WEIGHT * evidence * (1 - conf) + CUT_BACKGROUND_PRIOR).astype(np.float64)
    pairs = np.zeros((0, 2), dtype=np.int64)
    w = np.zeros(0, dtype=np.float64)
    xyz = state.means_cpu[candidates]
    col = state.colors_cpu[candidates]
    k = min(CUT_PAIRWISE_K, len(candidates) - 1)
    if k > 0:
        tree = cKDTree(xyz)
        d, ix = tree.query(xyz, k=k + 1, workers=-1)
        d = d[:, 1:].reshape(-1)
        ix = ix[:, 1:].reshape(-1)
        row = np.repeat(np.arange(len(candidates)), k)
        keep = d <= CUT_PAIRWISE_RADIUS
        row = row[keep]
        ix = ix[keep]
        d = d[keep]
        a = np.minimum(row, ix)
        b = np.maximum(row, ix)
        key = a.astype(np.int64) * len(candidates) + b
        _, pos = np.unique(key, return_index=True)
        pairs = np.stack([a[pos], b[pos]], 1)
        if len(pairs):
            sd = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
            cd = np.linalg.norm(col[pairs[:, 0]] - col[pairs[:, 1]], axis=1)
            w = CUT_PAIRWISE_WEIGHT * np.exp(
                -(sd**2) / (2 * CUT_PAIRWISE_SPATIAL_SIGMA**2)
                - (cd**2) / (2 * CUT_PAIRWISE_COLOR_SIGMA**2)
            )
    sink_side = min_cut(source, sink, pairs, w)
    return candidates[~sink_side]


# =============================================================================
# 6. SAM 3 PROMPTING AND MASK HELPERS
# =============================================================================


def load_sam3():
    """Load the SAM 3 image model and processor (requires Hugging Face access to the weights)."""
    if state.sam3_processor is not None:
        return state.sam3_processor

    model = build_sam3_image_model(device=DEVICE)
    state.sam3_processor = Sam3Processor(model, confidence_threshold=0.0)
    return state.sam3_processor


def new_sam_state(image):
    processor = load_sam3()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        return processor.set_image(image)


def extract_mask_near(result, cx_px, cy_px):
    masks = result.get("masks")
    scores = result.get("scores")
    if masks is None or scores is None or scores.numel() == 0:
        return None, 0.0
    masks = masks.detach().float().cpu().numpy()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    masks = masks.astype(bool)
    scores = scores.detach().float().cpu().numpy().astype(np.float32)
    cx = int(np.clip(round(cx_px), 0, state.WIDTH - 1))
    cy = int(np.clip(round(cy_px), 0, state.HEIGHT - 1))
    center = masks[:, cy, cx]
    if center.any():
        cand = np.flatnonzero(center)
    else:
        boxes = result.get("boxes")
        if boxes is None:
            return None, 0.0
        b = boxes.detach().float().cpu().numpy()
        bc = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
        cand = np.array([np.argmin(np.linalg.norm(bc - np.array([cx, cy]), axis=1))])
    best = cand[np.argmax(scores[cand])]
    mask = masks[best]
    labeled, _ = ndimage.label(mask)
    click_label = labeled[cy, cx]
    if click_label > 0:
        mask = labeled == click_label
    return mask, float(scores[best])


def extract_mask_top(result):
    masks = result.get("masks")
    scores = result.get("scores")
    if masks is None or scores is None or scores.numel() == 0:
        return None, 0.0
    masks = masks.detach().float().cpu().numpy()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    masks = masks.astype(bool)
    scores = scores.detach().float().cpu().numpy().astype(np.float32)
    best = int(np.argmax(scores))
    mask = masks[best]
    labeled, num = ndimage.label(mask)
    if num > 1:
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
        mask = labeled == (1 + int(np.argmax(sizes)))
    return mask, float(scores[best])


def segment_point_at(sam_state, px, py):
    """Segment the object under a click with SAM 3's detector, using only its public API."""
    processor = load_sam3()
    processor.reset_all_prompts(sam_state)
    point = torch.tensor(
        [px / state.WIDTH, py / state.HEIGHT], device=processor.device, dtype=torch.float32
    ).view(1, 1, 2)
    label = torch.tensor([True], device=processor.device).view(1, 1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        # "visual" is SAM 3's text for geometry-only prompts. The first call sets up an empty
        # geometric prompt; the second runs the detector with the click added to it.
        processor.set_text_prompt("visual", sam_state)
        sam_state["geometric_prompt"].append_points(point, label)
        result = processor.set_text_prompt("visual", sam_state)
    return extract_mask_near(result, px, py)


def segment_text(sam_state, prompt):
    processor = load_sam3()
    # Drop any click point left in the state, so the text prompt runs on its own.
    processor.reset_all_prompts(sam_state)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        result = processor.set_text_prompt(prompt, sam_state)
    return extract_mask_top(result)


def grow_mask(mask, px):
    if mask is None or px <= 0:
        return mask
    mask = ndimage.binary_fill_holes(mask)
    return ndimage.distance_transform_edt(~mask) <= float(px)


def mask_preview_image(img, mask):
    a = np.asarray(img).astype(np.float32)
    o = a.copy()
    tint = np.array([255.0, 0.0, 255.0])
    o[mask] = 0.45 * o[mask] + 0.55 * tint
    return Image.fromarray(o.astype(np.uint8))


# =============================================================================
# 7. OBJECT EXPORT
# =============================================================================


def write_object_ply(source_ply, gaussian_indices, output_path, recenter=True):
    """Write the selected Gaussians from `source_ply` to a new .ply file.

    With `recenter`, the Gaussians are moved so their bounding-box centre is at the origin;
    otherwise they keep their original scene positions.
    """
    indices = np.asarray(gaussian_indices, dtype=np.int64)
    if indices.ndim != 1 or not indices.size:
        raise ValueError("Cannot export an object with no Gaussian vertices.")
    if np.any(indices < 0) or np.any(indices >= len(source_ply.vertices)):
        raise ValueError("Object Gaussian indices do not match the loaded PLY.")
    object_data = source_ply.vertices[indices]
    if recenter:
        xyz = np.stack([object_data["x"], object_data["y"], object_data["z"]], 1).astype(np.float64)
        pivot = (xyz.min(0) + xyz.max(0)) / 2.0
        for i, axis in enumerate("xyz"):
            object_data[axis] = (xyz[:, i] - pivot[i]).astype(object_data.dtype[axis])
    write_ply(
        output_path,
        object_data,
        source_ply.byte_order,
        source_ply.comments,
        source_ply.vertex_comments,
        source_ply.obj_info,
    )
