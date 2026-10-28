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

"""Configuration and shared runtime state for the interactive segmentation app.

Section 1 holds constants that never change at runtime; other modules may import
them by name (``from state import CUT_UNARY_WEIGHT``).

Section 2 holds values that are reassigned while the app runs. Always access these
through the module (``import state`` then ``state.means``); ``from state import means``
would copy the value once and never see later updates.
"""

# Standard library
import threading
from pathlib import Path

# Third-party
import numpy as np
import torch

# =============================================================================
# 1. CONFIGURATION (constant at runtime)
# =============================================================================


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRADIO_PORT = 7866
RENDER_HEIGHT = 600

# Segmentation / graph-cut tuning
SPATIAL_COMPONENT_RADIUS = 0.05
SEED_CONFIDENCE_THRESHOLD = 0.5
MIN_OBJECT_GAUSSIANS = 30
MIN_COMPONENT_SIZE = 8
COMPONENT_ATTACH_RADIUS = 0.15
# A selected Gaussian is dropped as an isolated stray when fewer than ISOLATED_MIN_SELECTED
# of its ISOLATED_NEIGHBOURS_K nearest scene neighbours are selected too.
ISOLATED_NEIGHBOURS_K = 8
ISOLATED_MIN_SELECTED = 2
CUT_BAND_RADIUS = 0.30
CUT_BAND_FRACTION = 0.5
VOTE_ANCHOR_THRESHOLD = 0.5
CUT_MAX_EXTENT_MULTIPLIER = 3.0
CUT_PAIRWISE_RADIUS = 0.05
CUT_PAIRWISE_K = 6
CUT_UNARY_WEIGHT = 20.0
CUT_PAIRWISE_WEIGHT = 2.0
CUT_PAIRWISE_SPATIAL_SIGMA = CUT_PAIRWISE_RADIUS / 2.0
CUT_PAIRWISE_COLOR_SIGMA = 0.15
# Gaussians seen at least this fraction as clearly as the median seed get the full data term;
# less visible ones get a proportionally weaker one, and unseen ones none. Kept low so that
# small or faint Gaussians inside the masks keep their full vote.
CUT_EVIDENCE_SATURATION = 0.02
# Per-Gaussian bias towards background. Lower lets the object grow further into unseen
# Gaussians through similar neighbours; higher keeps it closer to what the masks show.
# It adds up over every unseen Gaussian, so it must stay small for dense splats; 0 leaks
# into hidden neighbouring objects.
CUT_BACKGROUND_PRIOR = 0.005
MAX_CUT_CANDIDATES = 300000
# SciPy's max-flow needs integer capacities: weights are multiplied by this and rounded.
CUT_FLOW_SCALE = 1e4
PENDING_HIGHLIGHT_COLOR = (1.0, 0.0, 1.0)
CLIP_DEBOUNCE_S = 0.3

# =============================================================================
# 2. RUNTIME STATE (access as state.<name>)
# =============================================================================


# Server settings, overridden from the command line in main()
# Exports go to <project root>/outputs/<scene name>/, no matter where the app is started from.
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
HOST = "127.0.0.1"
VISER_PORT = 8080
viser_server = None  # created by start_viser_server()

# Camera intrinsics of the last captured view, updated by capture_view()
WIDTH, HEIGHT = 800, 600
FX, FY = 700.0, 700.0
CX, CY = WIDTH / 2.0, HEIGHT / 2.0
focal_px = (FX + FY) / 2.0
K = torch.tensor(
    [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], device=DEVICE, dtype=torch.float32
)

# Scene and segmentation session
means = quats = scales = opacities = colors = None
means_cpu = colors_cpu = full_tree = None
claimed = None
N = 0
loaded_ply_path = None
sam3_processor = None
current_viewmat = None
current_base_image = None
current_camera_pose = None
pending_views = []
pending_mask_candidate = None
pending_sam_state = None
object_records = []
next_object_id = 1
scene_center = np.zeros(3, dtype=np.float32)
scene_radius = 3.0
splat_node = None
active_client_id = None
preview_viewmat = None
preview_width = preview_height = None
preview_k = None

# Clipping box
clip_enabled = False
clip_center = clip_half = None
clip_rot = np.eye(3, dtype=np.float32)
clip_wxyz = (1.0, 0.0, 0.0, 0.0)
clip_mask = None
splat_cpu = None
clip_box_node = None
clip_gui = None
clip_syncing = False
clip_timer = None
clip_lock = threading.RLock()
