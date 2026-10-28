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
import argparse
import re
from pathlib import Path

# Third-party
import gradio as gr
import numpy as np
import torch
from scipy.spatial import cKDTree

# Local
import state
from ply_io import read_ply
from segmentation import (
    accumulate_all_pending_views,
    build_display_colors,
    drop_isolated,
    graph_cut_object,
    grow_mask,
    load_ply,
    load_sam3,
    load_splat,
    log_exception,
    mask_preview_image,
    new_sam_state,
    quats_scales_to_covariances,
    render_preview,
    render_view,
    segment_point_at,
    segment_text,
    spatial_connected_components,
    viewmat_from_camera_pose,
    warmup_gsplat,
    write_object_ply,
)
from state import (
    COMPONENT_ATTACH_RADIUS,
    DEVICE,
    GRADIO_PORT,
    ISOLATED_MIN_SELECTED,
    ISOLATED_NEIGHBOURS_K,
    MIN_COMPONENT_SIZE,
    MIN_OBJECT_GAUSSIANS,
    RENDER_HEIGHT,
    SEED_CONFIDENCE_THRESHOLD,
    SPATIAL_COMPONENT_RADIUS,
)
from viewer import (
    get_viser_client,
    refresh_viewer_splats,
    setup_clip_gui,
    start_viser_server,
    viser_camera_pose_dict,
)

# =============================================================================
# 1. GRADIO EVENT HANDLERS
# =============================================================================


def build_objects_table():
    return [
        [r["object_id"], r["label"], int(r["gaussian_indices"].size)] for r in state.object_records
    ]


def on_load_scene(file_path):
    if file_path is None:
        return None, None, "No file selected."
    viewer_path, message = load_splat(file_path)
    if viewer_path is None:
        return None, None, message
    try:
        selected_path = Path(viewer_path)
        if selected_path.suffix.lower() != ".ply":
            return None, None, "The segmentation backend currently requires a .ply file."
        state.means, state.quats, state.scales, state.opacities, state.colors = load_ply(
            selected_path
        )
        state.N = state.means.shape[0]
        state.means_cpu = state.means.detach().cpu().numpy().astype(np.float32, copy=False)
        state.colors_cpu = state.colors.detach().cpu().numpy().astype(np.float32, copy=False)
        opacities_cpu = state.opacities.detach().cpu().numpy().astype(np.float32, copy=False)
        quats_cpu = state.quats.detach().cpu().numpy().astype(np.float32, copy=False)
        scales_cpu = state.scales.detach().cpu().numpy().astype(np.float32, copy=False)
        state.full_tree = cKDTree(state.means_cpu)
        state.claimed = np.zeros(state.N, dtype=bool)
        state.loaded_ply_path = selected_path
        state.object_records = []
        state.next_object_id = 1
        state.current_base_image = None
        state.current_viewmat = None
        state.current_camera_pose = None
        state.pending_views.clear()
        state.pending_mask_candidate = None
        state.pending_sam_state = None
        state.scene_center = state.means_cpu.mean(axis=0).astype(np.float32)
        extent = np.linalg.norm(state.means_cpu - state.scene_center, axis=1)
        state.scene_radius = max(float(np.percentile(extent, 95)) * 2.5, 1.0)
        covariances = quats_scales_to_covariances(quats_cpu, scales_cpu)
        state.splat_cpu = {
            "centers": state.means_cpu,
            "rgbs": state.colors_cpu,
            "opacities": opacities_cpu[:, None],
            "covariances": covariances,
        }
        setup_clip_gui()
        refresh_viewer_splats()
        return (
            None,
            None,
            f"{message}\nBackend loaded: {state.N:,} Gaussians.\nCamera center: {state.scene_center.tolist()}\nOpen the 3D panel on the right, navigate to a view, then click Capture View.",
        )
    except Exception as e:
        log_exception("Backend loading failed")
        return None, None, f"Backend loading failed: {e}"


def capture_view():
    if state.means is None:
        return None, "Load a PLY scene first."
    client = get_viser_client()
    if client is None:
        return (
            None,
            "No 3D viewer connected yet. Open/reload the panel on the right, navigate to a view, then click Capture View again.",
        )
    try:
        aspect = float(client.camera.aspect)
        fov_y = float(client.camera.fov)
        state.HEIGHT = RENDER_HEIGHT
        state.WIDTH = max(int(round(state.HEIGHT * aspect)), 1)
        state.FY = state.HEIGHT / (2.0 * np.tan(fov_y / 2.0))
        state.FX = state.FY
        state.CX = state.WIDTH / 2.0
        state.CY = state.HEIGHT / 2.0
        state.focal_px = (state.FX + state.FY) / 2.0
        state.K = torch.tensor(
            [[state.FX, 0.0, state.CX], [0.0, state.FY, state.CY], [0.0, 0.0, 1.0]],
            device=DEVICE,
            dtype=torch.float32,
        )
        state.current_viewmat = viewmat_from_camera_pose(client.camera.position, client.camera.wxyz)
        state.current_base_image = render_view(state.current_viewmat, build_display_colors())
        state.current_camera_pose = viser_camera_pose_dict(client)
        state.pending_mask_candidate = None
        state.pending_sam_state = None
        pos = state.current_camera_pose["position"]
        progress = (
            f"\n{len(state.pending_views)} view(s) saved with masks so far."
            if state.pending_views
            else "\nClick a point on the object (or use a text prompt), then refine and Add Mask."
        )
        return (
            state.current_base_image,
            f"View captured.\nCamera XYZ: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]{progress}",
        )
    except Exception:
        log_exception("Capture failed")
        return None, "Capture failed. Check terminal."


def on_image_click(mode, grow_px, evt: gr.SelectData):
    if state.means is None:
        return state.current_base_image, "Load a scene first."
    if state.current_viewmat is None:
        return state.current_base_image, "Capture a view first."
    if evt is None or evt.index is None:
        return state.current_base_image, "Click failed. Try again."
    px, py = evt.index
    add = mode == "Add (+)"
    try:
        if state.pending_sam_state is None:
            state.pending_sam_state = new_sam_state(state.current_base_image)
        click_mask, score = segment_point_at(state.pending_sam_state, px, py)
        if click_mask is None:
            return state.current_base_image, "SAM3 found no usable mask at that point."
        click_mask = grow_mask(click_mask, grow_px)
        first_click = state.pending_mask_candidate is None
        if first_click:
            combined = click_mask
        elif add:
            combined = state.pending_mask_candidate["mask"] | click_mask
        else:
            combined = state.pending_mask_candidate["mask"] & ~click_mask
        preview = mask_preview_image(state.current_base_image, combined)
        state.pending_mask_candidate = {
            "viewmat": state.current_viewmat,
            "mask": combined,
            "score": score,
            "width": state.WIDTH,
            "height": state.HEIGHT,
            "fx": state.FX,
            "fy": state.FY,
            "cx": state.CX,
            "cy": state.CY,
            "pose": state.current_camera_pose,
        }
        action = "Added" if (add or first_click) else "Removed"
        return (
            preview,
            f"{action} the clicked region (click score {score:.2f}). Click more points to keep refining, or Add Mask / Discard Mask.",
        )
    except Exception:
        log_exception("Point click failed")
        return state.current_base_image, "Point click failed. Check terminal."


def on_apply_text_prompt(prompt_text, grow_px):
    if state.means is None:
        return state.current_base_image, "Load a scene first."
    if state.current_viewmat is None:
        return state.current_base_image, "Capture a view first."
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        return state.current_base_image, "Enter a text prompt first."
    try:
        if state.pending_sam_state is None:
            state.pending_sam_state = new_sam_state(state.current_base_image)
        mask, score = segment_text(state.pending_sam_state, prompt_text)
        if mask is None:
            return state.current_base_image, f"SAM3 found no usable mask for '{prompt_text}'."
        mask = grow_mask(mask, grow_px)
        preview = mask_preview_image(state.current_base_image, mask)
        state.pending_mask_candidate = {
            "viewmat": state.current_viewmat,
            "mask": mask,
            "score": score,
            "width": state.WIDTH,
            "height": state.HEIGHT,
            "fx": state.FX,
            "fy": state.FY,
            "cx": state.CX,
            "cy": state.CY,
            "pose": state.current_camera_pose,
        }
        return (
            preview,
            f"Mask ready from text prompt '{prompt_text}' (score {score:.2f}). Click points to Add/Remove regions, or Add Mask to keep it, or Discard Mask to start over.\nNote: a text prompt matches the single highest-confidence object in view — if several similar objects are visible, click a point instead to pick the right one.",
        )
    except Exception:
        log_exception("Text-prompt mask creation failed")
        return state.current_base_image, "Text-prompt mask creation failed. Check terminal."


def on_add_mask():
    if state.pending_mask_candidate is None:
        return state.current_base_image, "No mask to add. Click a point or use a text prompt first."
    state.pending_views.append(state.pending_mask_candidate)
    state.pending_mask_candidate = None
    state.pending_sam_state = None
    return (
        state.current_base_image,
        f"Mask added. {len(state.pending_views)} view(s) saved.\nCapture another angle to add more, or click Segment to fuse {len(state.pending_views)} view(s) into an object.",
    )


def on_discard_mask():
    state.pending_mask_candidate = None
    state.pending_sam_state = None
    return (
        state.current_base_image,
        "Mask discarded. Click a new point or try a different text prompt.",
    )


def on_segment(isolate):
    if not state.pending_views:
        return (
            state.current_base_image,
            None,
            "No views saved yet. Capture a view, click the object, and Add Mask first.",
        )
    try:
        numer_np, denom_np = accumulate_all_pending_views()
        has_evidence = denom_np > 1e-6
        conf_all = np.zeros(state.N, np.float32)
        conf_all[has_evidence] = np.clip(numer_np[has_evidence] / denom_np[has_evidence], 0, 1)
        strong = np.flatnonzero(
            has_evidence & (conf_all > SEED_CONFIDENCE_THRESHOLD) & (~state.claimed)
        )
        if not strong.size:
            return (
                state.current_base_image,
                None,
                f"No confident Gaussians found from {len(state.pending_views)} view(s). Try clicking more precisely or capturing a closer view.",
            )
        comp = spatial_connected_components(state.means_cpu[strong], SPATIAL_COMPONENT_RADIUS)
        seed = strong[max(comp, key=len)]
        final = graph_cut_object(seed, numer_np, denom_np)
        if len(final) < MIN_OBJECT_GAUSSIANS:
            return (
                state.current_base_image,
                None,
                f"Selection too small: {len(final)} Gaussians from {len(state.pending_views)} view(s). Capture more angles and Apply, or discard.",
            )
        comps = spatial_connected_components(state.means_cpu[final], SPATIAL_COMPONENT_RADIUS)
        dropped = 0
        if comps:
            main = max(comps, key=len)
            main_tree = cKDTree(state.means_cpu[final[main]])
            kept = [main]
            for c in comps:
                if c is main or len(c) < MIN_COMPONENT_SIZE:
                    continue
                d, _ = main_tree.query(state.means_cpu[final[c]], k=1)
                if d.min() <= COMPONENT_ATTACH_RADIUS:
                    kept.append(c)
            dropped = len(final) - sum(len(c) for c in kept)
            final = final[np.sort(np.concatenate(kept))]
        final, isolated = drop_isolated(final, ISOLATED_NEIGHBOURS_K, ISOLATED_MIN_SELECTED)
        dropped += len(isolated)
        state.preview_viewmat, state.preview_width, state.preview_height, state.preview_k = (
            state.current_viewmat,
            state.WIDTH,
            state.HEIGHT,
            state.K,
        )
        selected = render_preview(final, isolate)
        note = (
            f" ({dropped} stray/floater Gaussians dropped as disconnected from or isolated around the main body.)"
            if dropped
            else ""
        )
        return (
            selected,
            final,
            f"Selection ready: {len(final):,} Gaussians fused from {len(state.pending_views)} view(s) via per-Gaussian radiance-weight fusion.{note} Enter a label and confirm, navigate the 3D panel and click Refresh Preview to inspect other sides, or discard.",
        )
    except Exception:
        log_exception("Segmentation failed")
        return state.current_base_image, None, "Segmentation failed. Check terminal."


def on_refresh_preview(pending, isolate):
    if pending is None or not len(pending):
        return state.current_base_image, "No selection yet. Run Segment first."
    client = get_viser_client()
    if client is None:
        return (
            state.current_base_image,
            "No 3D viewer connected yet. Open/reload the panel on the right.",
        )
    try:
        aspect = float(client.camera.aspect)
        fov_y = float(client.camera.fov)
        h = RENDER_HEIGHT
        w = max(int(round(h * aspect)), 1)
        fy = h / (2.0 * np.tan(fov_y / 2.0))
        fx = fy
        cx = w / 2.0
        cy = h / 2.0
        k = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=DEVICE, dtype=torch.float32
        )
        viewmat = viewmat_from_camera_pose(client.camera.position, client.camera.wxyz)
        state.preview_viewmat, state.preview_width, state.preview_height, state.preview_k = (
            viewmat,
            w,
            h,
            k,
        )
        preview = render_preview(pending, isolate)
        return (
            preview,
            "Preview refreshed from the current 3D viewer angle. Navigate the 3D panel further and Refresh Preview again to check other sides.",
        )
    except Exception:
        log_exception("Refresh preview failed")
        return state.current_base_image, "Refresh preview failed. Check terminal."


def on_toggle_isolate(pending, isolate):
    if pending is None or not len(pending):
        return state.current_base_image, "No selection yet. Run Segment first."
    if state.preview_viewmat is None:
        return state.current_base_image, "Run Segment (or Refresh Preview) first."
    try:
        preview = render_preview(pending, isolate)
        msgs = {
            "isolate": "Showing only the segmented object.",
            "invert": "Showing the whole scene without the segmented object.",
        }
        return preview, msgs.get(
            isolate, "Showing the segmented object highlighted within the full scene."
        )
    except Exception:
        log_exception("Toggle isolate view failed")
        return state.current_base_image, "Toggle failed. Check terminal."


def on_confirm(label, pending):
    if pending is None or not len(pending):
        return (
            state.current_base_image,
            state.current_base_image,
            build_objects_table(),
            None,
            "",
            "Nothing selected.",
        )
    label = (label or "").strip()
    if not label:
        return (
            state.current_base_image,
            state.current_base_image,
            build_objects_table(),
            pending,
            label,
            "Enter a label.",
        )
    try:
        oid = state.next_object_id
        state.next_object_id += 1
        state.claimed[pending] = True
        state.object_records.append(
            {
                "object_id": oid,
                "label": label,
                "gaussian_indices": pending.copy(),
                "camera_poses": [v["pose"] for v in state.pending_views],
            }
        )
        state.current_base_image = render_view(state.current_viewmat, build_display_colors())
        view_count = len(state.pending_views)
        state.pending_views.clear()
        state.pending_mask_candidate = None
        state.pending_sam_state = None
        state.preview_viewmat = None
        return (
            state.current_base_image,
            state.current_base_image,
            build_objects_table(),
            None,
            "",
            f"Confirmed object {oid}: {label} (from {view_count} view(s)).",
        )
    except Exception:
        log_exception("Confirm failed")
        return (
            state.current_base_image,
            state.current_base_image,
            build_objects_table(),
            pending,
            label,
            "Confirm failed.",
        )


def on_discard():
    state.pending_views.clear()
    state.pending_mask_candidate = None
    state.pending_sam_state = None
    state.preview_viewmat = None
    return state.current_base_image, state.current_base_image, None, "Selection discarded."


def on_undo():
    if not state.object_records:
        return (
            state.current_base_image,
            state.current_base_image,
            build_objects_table(),
            "Nothing to undo.",
        )
    r = state.object_records.pop()
    state.claimed[r["gaussian_indices"]] = False
    state.current_base_image = render_view(state.current_viewmat, build_display_colors())
    return (
        state.current_base_image,
        state.current_base_image,
        build_objects_table(),
        f"Removed object {r['object_id']}.",
    )


def on_reset():
    if state.means is None:
        return state.current_base_image, state.current_base_image, [], "Load a scene first."
    state.object_records = []
    state.next_object_id = 1
    state.claimed[:] = False
    state.pending_views.clear()
    state.pending_mask_candidate = None
    state.pending_sam_state = None
    state.preview_viewmat = None
    if state.current_viewmat is not None:
        state.current_base_image = render_view(state.current_viewmat)
    return state.current_base_image, state.current_base_image, [], "All labels reset."


def on_save():
    if not state.object_records:
        return "No labeled objects to save."
    try:
        if state.loaded_ply_path is None:
            raise RuntimeError("Load a PLY scene before exporting objects.")
        source_ply = read_ply(state.loaded_ply_path)
        if len(source_ply.vertices) != state.N:
            raise ValueError(
                "The loaded PLY vertex count changed; reload the scene before exporting."
            )
        scene_dir = state.OUT_DIR / Path(state.loaded_ply_path).stem
        objects_dir = scene_dir / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)
        ids = np.zeros(state.N, dtype=np.uint32)
        ply_paths = []
        for r in state.object_records:
            ids[r["gaussian_indices"]] = r["object_id"]
            label_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", r["label"]).strip("_")[:64] or "object"
            ply_path = objects_dir / f"object_{r['object_id']}_{label_slug}.ply"
            write_object_ply(source_ply, r["gaussian_indices"], ply_path)
            ply_paths.append(ply_path)
        ids_path = scene_dir / "gaussian_object_ids.bin"
        ids.tofile(ids_path)
        return (
            f"Saved {len(state.object_records)} objects.\nIDs: {ids_path}\nObject PLYs:\n"
            + "\n".join(str(path) for path in ply_paths)
        )
    except Exception as e:
        log_exception("Save failed")
        return f"Save failed: {e}"


def on_export_scene_without_objects():
    if not state.object_records:
        return "No labeled objects to remove. Confirm at least one object first."
    try:
        if state.loaded_ply_path is None:
            raise RuntimeError("Load a PLY scene before exporting.")
        source_ply = read_ply(state.loaded_ply_path)
        if len(source_ply.vertices) != state.N:
            raise ValueError(
                "The loaded PLY vertex count changed; reload the scene before exporting."
            )
        removed = np.zeros(state.N, dtype=bool)
        for r in state.object_records:
            removed[r["gaussian_indices"]] = True
        remaining = np.flatnonzero(~removed)
        if not remaining.size:
            return "The labeled objects cover the whole scene; nothing is left to export."
        scene_name = Path(state.loaded_ply_path).stem
        scene_dir = state.OUT_DIR / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        out_path = scene_dir / f"{scene_name}_without_objects.ply"
        write_object_ply(source_ply, remaining, out_path, recenter=False)
        return (
            f"Exported the scene without {len(state.object_records)} labeled object(s): "
            f"{remaining.size:,} of {state.N:,} Gaussians kept "
            f"({int(removed.sum()):,} removed).\n{out_path}"
        )
    except Exception as e:
        log_exception("Scene export failed")
        return f"Scene export failed: {e}"


# =============================================================================
# 2. GRADIO UI LAYOUT
# =============================================================================


LOGO_IMAGE_URL = "https://cli.filmakademie.de/files/public/images/04-forschung/CLI_FABW.png"
LOGO_URL = "https://cli.filmakademie.de"
# The logo is white, so it needs a dark background.
LOGO_CSS = (
    "#logo { background: #1a1a1a; border-radius: 8px; padding: 16px; }"
    " #logo, #logo * { cursor: pointer; }"
    # Sticky footer: let the page fill the window and push the logo to the bottom.
    " main.contain, main.contain > div:has(#logo) { flex-grow: 1; }"
    " #logo { margin-top: auto; }"
)
# Default to dark mode via Gradio's ?__theme= URL parameter
DARK_MODE_JS = """
() => {
    const url = new URL(window.location);
    if (!url.searchParams.has("__theme")) {
        url.searchParams.set("__theme", "dark");
        window.location.replace(url);
    }
}
"""


def build_ui():
    """Build the Gradio interface."""
    viewer_host = "127.0.0.1" if state.HOST in ("0.0.0.0", "") else state.HOST
    with gr.Blocks(title="SegmentThat: Interactive Gaussian Splat Segmentation") as demo:
        gr.Markdown(
            f"# SegmentThat: Interactive Gaussian Splat Segmentation\nLoad a Gaussian splat, then for each object: navigate in the 3D panel on the right and **Capture View**, then click a point directly on the object (and/or type a **Text Prompt** and click Apply Text Prompt) to get a candidate SAM3 mask. Set the mode to **Add (+)** or **Remove (-)** and keep clicking points on the image to refine the same mask e.g. click a missing part to add it, or a wrongly-included part to remove it. When it looks right, **Add Mask** to save this view; if it's a lost cause, **Discard Mask** and start over. Repeat Capture+mask from as many other angles as you like, then click **Segment** to fuse all saved views into a 3D selection (re-runnable: add more views and Segment again to refine). When it looks right, enter a label and **Confirm & Label**. **Discard Selection** clears all saved views for the in-progress object and starts over. **Save to Disk** writes each labeled object as a separate PLY in `outputs/<scene>/objects/`. **Export Scene without Objects** writes the loaded scene with all labeled objects removed, at its original positions, to `outputs/<scene>/`."
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Gaussian Splat File", file_types=[".ply"], type="filepath"
                )
                load_btn = gr.Button("Load Scene", variant="primary")
                capture_btn = gr.Button("Capture View", variant="primary")
                status = gr.Textbox(label="Status", lines=7, interactive=False)

            with gr.Column(scale=2):
                gr.HTML(
                    f'<iframe src="http://{viewer_host}:{state.VISER_PORT}" style="width:100%;height:700px;border:1px solid #444;"></iframe>'
                )

        with gr.Row():
            with gr.Column(scale=2):
                captured_image = gr.Image(
                    label="Captured View - click a point on the object to add/remove it from the mask",
                    type="numpy",
                    interactive=False,
                )
                with gr.Row():
                    click_mode = gr.Radio(
                        choices=["Add (+)", "Remove (-)"],
                        value="Add (+)",
                        label="Click mode",
                        scale=2,
                    )
                    text_prompt_box = gr.Textbox(
                        label="Text Prompt (alternative to clicking)",
                        placeholder="e.g. 'chair', 'red mug'",
                        scale=3,
                    )
                    text_apply_btn = gr.Button("Apply Text Prompt", scale=1)
                with gr.Accordion("Grow mask edges", open=False):
                    grow_slider = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=1,
                        step=1,
                        label="Pixels (applied to each new SAM mask; increase if edges are cut off)",
                    )
            with gr.Column(scale=1):
                with gr.Row():
                    add_mask_btn = gr.Button("Add Mask", variant="primary")
                    discard_mask_btn = gr.Button("Discard Mask")
                segment_btn = gr.Button("Segment (fuse all saved views)", variant="primary")
                refresh_preview_btn = gr.Button("Refresh Preview (from current 3D view)")
                isolate_toggle = gr.Radio(
                    choices=[
                        ("Highlight object in scene", "highlight"),
                        ("Show only segmented object", "isolate"),
                        ("Show scene without segmented object", "invert"),
                    ],
                    value="highlight",
                    label="Preview mode",
                )
                preview_image = gr.Image(label="3D Gaussian Selection", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                label_box = gr.Textbox(label="Object label", placeholder="chair, table, wall")
                confirm_btn = gr.Button("Confirm & Label", variant="primary")
                discard_btn = gr.Button("Discard Selection")
                undo_btn = gr.Button("Undo Last Object")
                reset_btn = gr.Button("Reset All Labels")
                save_btn = gr.Button("Save to Disk (IDs & object PLYs)", variant="stop")
                export_scene_btn = gr.Button("Export Scene without Objects (PLY)")
            with gr.Column(scale=2):
                objects_table = gr.Dataframe(
                    headers=["id", "label", "gaussian_count"],
                    label="Labeled Objects",
                    interactive=False,
                )

        logo = gr.Image(
            value=LOGO_IMAGE_URL,
            type="filepath",
            height=100,
            show_label=False,
            interactive=False,
            container=False,
            buttons=[],
            elem_id="logo",
        )
        logo.select(fn=None, js=f"() => {{ window.open('{LOGO_URL}', '_blank'); }}")

        pending_indices = gr.State(None)

        load_btn.click(
            fn=on_load_scene,
            inputs=file_input,
            outputs=[captured_image, preview_image, status],
            show_progress="full",
        )

        capture_btn.click(fn=capture_view, inputs=[], outputs=[captured_image, status])

        captured_image.select(
            fn=on_image_click, inputs=[click_mode, grow_slider], outputs=[captured_image, status]
        )

        text_apply_btn.click(
            fn=on_apply_text_prompt,
            inputs=[text_prompt_box, grow_slider],
            outputs=[captured_image, status],
        )

        add_mask_btn.click(fn=on_add_mask, inputs=[], outputs=[captured_image, status])

        discard_mask_btn.click(fn=on_discard_mask, inputs=[], outputs=[captured_image, status])

        segment_btn.click(
            fn=on_segment, inputs=[isolate_toggle], outputs=[preview_image, pending_indices, status]
        )

        refresh_preview_btn.click(
            fn=on_refresh_preview,
            inputs=[pending_indices, isolate_toggle],
            outputs=[preview_image, status],
        )

        isolate_toggle.change(
            fn=on_toggle_isolate,
            inputs=[pending_indices, isolate_toggle],
            outputs=[preview_image, status],
        )

        confirm_btn.click(
            fn=on_confirm,
            inputs=[label_box, pending_indices],
            outputs=[
                captured_image,
                preview_image,
                objects_table,
                pending_indices,
                label_box,
                status,
            ],
        )

        discard_btn.click(
            fn=on_discard, outputs=[captured_image, preview_image, pending_indices, status]
        )

        undo_btn.click(fn=on_undo, outputs=[captured_image, preview_image, objects_table, status])

        reset_btn.click(fn=on_reset, outputs=[captured_image, preview_image, objects_table, status])

        save_btn.click(fn=on_save, outputs=status)

        export_scene_btn.click(fn=on_export_scene_without_objects, outputs=status)
    return demo


# =============================================================================
# 3. ENTRY POINT
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="SegmentThat: Interactive Gaussian Splat Segmentation."
    )
    parser.add_argument("--host", default=state.HOST, help="Host for the Gradio and Viser servers.")
    parser.add_argument("--port", type=int, default=GRADIO_PORT, help="Gradio UI port.")
    parser.add_argument("--viser-port", type=int, default=state.VISER_PORT, help="3D viewer port.")
    parser.add_argument(
        "--out-dir", type=Path, default=state.OUT_DIR, help="Folder for saved results."
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    return parser.parse_args()


def main():
    args = parse_args()
    state.HOST, state.VISER_PORT, state.OUT_DIR = args.host, args.viser_port, args.out_dir
    start_viser_server(state.HOST, state.VISER_PORT)
    warmup_gsplat()
    load_sam3()
    build_ui().launch(
        server_name=state.HOST,
        server_port=args.port,
        share=args.share,
        css=LOGO_CSS,
        js=DARK_MODE_JS,
    )


if __name__ == "__main__":
    main()
