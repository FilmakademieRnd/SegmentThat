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
import threading

# Third-party
import numpy as np
import torch
import viser
from scipy.spatial.transform import Rotation

# Local
import state
from segmentation import log
from state import CLIP_DEBOUNCE_S, DEVICE

# =============================================================================
# 1. VIEWER SERVER
# =============================================================================


def on_viser_client_connect(client):
    state.active_client_id = client.client_id
    log(
        f"Viser client connected: id={client.client_id}, total clients={len(state.viser_server.get_clients())}"
    )


def on_viser_client_disconnect(client):
    log(f"Viser client disconnected: id={client.client_id}")
    if state.active_client_id == client.client_id:
        remaining = state.viser_server.get_clients()
        state.active_client_id = next(iter(remaining.keys())) if remaining else None


def start_viser_server(host, port):
    """Start the Viser viewer server and register client tracking callbacks."""
    state.viser_server = viser.ViserServer(host=host, port=port)
    state.viser_server.gui.configure_theme(dark_mode=True)
    state.viser_server.on_client_connect(on_viser_client_connect)
    state.viser_server.on_client_disconnect(on_viser_client_disconnect)
    return state.viser_server


# =============================================================================
# 2. VISER CLIENT CAMERA
# =============================================================================


def get_viser_client():
    clients = state.viser_server.get_clients()
    if not clients:
        return None
    if state.active_client_id is not None and state.active_client_id in clients:
        return clients[state.active_client_id]
    return next(iter(clients.values()))


def viser_camera_pose_dict(client):
    return {
        "position": [float(v) for v in client.camera.position],
        "wxyz": [float(v) for v in client.camera.wxyz],
        "fov": float(client.camera.fov),
    }


# =============================================================================
# 3. CLIPPING BOX
# =============================================================================


def update_clip_mask():
    """Recompute the per-Gaussian mask for the current clipping box."""
    if not state.clip_enabled or state.means is None:
        state.clip_mask = None
        return
    c = torch.tensor(state.clip_center, device=DEVICE, dtype=state.means.dtype)
    h = torch.tensor(state.clip_half, device=DEVICE, dtype=state.means.dtype)
    r = torch.tensor(state.clip_rot, device=DEVICE, dtype=state.means.dtype)
    local = (state.means - c) @ r
    state.clip_mask = (local.abs() <= h).all(dim=1).to(state.opacities.dtype)


def clip_inside_np(points):
    return (np.abs((points - state.clip_center) @ state.clip_rot) <= state.clip_half).all(1)


def update_clip_box_node():
    with state.clip_lock:
        if not state.clip_enabled:
            if state.clip_box_node is not None:
                try:
                    state.clip_box_node.remove()
                except Exception:
                    pass
                state.clip_box_node = None
            return
        dims = tuple(float(v) for v in state.clip_half * 2)
        pos = tuple(float(v) for v in state.clip_center)
        if state.clip_box_node is None:
            state.clip_box_node = state.viser_server.scene.add_box(
                "/clip_box",
                color=(255, 200, 0),
                dimensions=dims,
                position=pos,
                wxyz=state.clip_wxyz,
                wireframe=True,
            )
        else:
            state.clip_box_node.dimensions = dims
            state.clip_box_node.position = pos
            state.clip_box_node.wxyz = state.clip_wxyz


def refresh_viewer_splats():
    """Re-upload the (clipped) splats to the Viser scene."""
    if state.splat_cpu is None:
        return
    with state.clip_lock:
        if state.splat_node is not None:
            try:
                state.splat_node.remove()
            except Exception:
                pass
            state.splat_node = None
        keep = np.ones(len(state.splat_cpu["centers"]), dtype=bool)
        if state.clip_enabled:
            keep = clip_inside_np(state.splat_cpu["centers"])
        update_clip_box_node()
        if not keep.any():
            return
        state.splat_node = state.viser_server.scene.add_gaussian_splats(
            "/scene",
            centers=state.splat_cpu["centers"][keep],
            rgbs=state.splat_cpu["rgbs"][keep],
            opacities=state.splat_cpu["opacities"][keep],
            covariances=state.splat_cpu["covariances"][keep],
        )


def commit_clip():
    with state.clip_lock:
        update_clip_mask()
        refresh_viewer_splats()
        n_in = int(state.clip_mask.sum().item()) if state.clip_mask is not None else state.N
    log(
        f"Clipping {'on' if state.clip_enabled else 'off'}: {n_in:,}/{state.N:,} Gaussians inside box"
    )


def schedule_clip_commit():
    with state.clip_lock:
        if state.clip_timer is not None:
            state.clip_timer.cancel()
        state.clip_timer = threading.Timer(CLIP_DEBOUNCE_S, commit_clip)
        state.clip_timer.daemon = True
        state.clip_timer.start()


def set_clip_from_center_size(center, size, wxyz=None):
    state.clip_center = np.asarray(center, dtype=np.float32)
    state.clip_half = np.maximum(np.asarray(size, dtype=np.float32), 1e-4) / 2.0
    if wxyz is not None:
        w, x, y, z = [float(v) for v in wxyz]
        state.clip_wxyz = (w, x, y, z)
        state.clip_rot = Rotation.from_quat([x, y, z, w]).as_matrix().astype(np.float32)


def apply_clip(*_, from_gizmo=False, commit="debounce"):
    if state.clip_gui is None or state.clip_syncing:
        return
    with state.clip_lock:
        state.clip_enabled = bool(state.clip_gui["enabled"].value)
        gizmo = state.clip_gui["gizmo"]
        set_clip_from_center_size(
            [s.value for s in state.clip_gui["center"]],
            [s.value for s in state.clip_gui["size"]],
            gizmo.wxyz,
        )
        gizmo.visible = state.clip_enabled
        if not from_gizmo:
            gizmo.position = tuple(float(v) for v in state.clip_center)
        update_clip_box_node()
        if state.clip_timer is not None:
            state.clip_timer.cancel()
    if commit == "now":
        commit_clip()
    elif commit == "debounce":
        schedule_clip_commit()


def on_gizmo_moved(event=None):
    if state.clip_gui is None:
        return
    pos = np.asarray(state.clip_gui["gizmo"].position, dtype=np.float32)
    state.clip_syncing = True
    try:
        for s, v in zip(state.clip_gui["center"], pos):
            s.value = float(v)
    finally:
        state.clip_syncing = False
    phase = getattr(event, "phase", None)
    apply_clip(
        from_gizmo=True, commit="now" if phase == "end" else "debounce" if phase is None else "none"
    )


def setup_clip_gui():
    if state.clip_gui is not None:
        handles = (
            [state.clip_gui["enabled"], state.clip_gui["gizmo"]]
            + state.clip_gui["center"]
            + state.clip_gui["size"]
        )
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        state.clip_gui = None
    state.clip_enabled = False
    lo = np.percentile(state.means_cpu, 0.5, axis=0).astype(np.float32)
    hi = np.percentile(state.means_cpu, 99.5, axis=0).astype(np.float32)
    pad = (hi - lo) * 0.1
    lo_b, hi_b = lo - pad, hi + pad
    center0 = (lo_b + hi_b) / 2
    size0 = hi_b - lo_b
    set_clip_from_center_size(center0, size0, (1.0, 0.0, 0.0, 0.0))
    state.clip_mask = None
    enabled = state.viser_server.gui.add_checkbox("Bounding box clipping", False)
    center, size = [], []
    for i, axis in enumerate("XYZ"):
        step = max(float(size0[i]) / 1000.0, 1e-4)
        center.append(
            state.viser_server.gui.add_slider(
                f"Center {axis}",
                min=float(lo_b[i]),
                max=float(hi_b[i]),
                step=step,
                initial_value=float(center0[i]),
            )
        )
    for i, axis in enumerate("XYZ"):
        step = max(float(size0[i]) / 1000.0, 1e-4)
        size.append(
            state.viser_server.gui.add_slider(
                f"Size {axis}",
                min=step,
                max=float(size0[i]) * 1.5,
                step=step,
                initial_value=float(size0[i]),
            )
        )
    gizmo = state.viser_server.scene.add_transform_controls(
        "/clip_gizmo",
        scale=float(size0.max()) * 0.05,
        position=tuple(float(v) for v in center0),
        disable_rotations=False,
        visible=False,
    )
    state.clip_gui = {"enabled": enabled, "center": center, "size": size, "gizmo": gizmo}
    for h in [enabled] + center + size:
        h.on_update(apply_clip)
    gizmo.on_update(on_gizmo_moved)
