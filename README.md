
# SegmentThat: Interactive Gaussian Splat Segmentation

**SegmentThat** is a tool that provides an interactive workflow for segmenting and exporting individual objects from 3D Gaussian Splatting (3DGS) scenes. It is developed in the context of the **Industrial Metaverse Reallabor** project, funded by the Ministry of Economics Baden-Württemberg ([project page](https://cli.filmakademie.de/en/research/projects/living-lab-metaverse)).

The workflow integrates multi-view 2D segmentation masks, differentiable-rendering-based foreground attribution, and 3D graph-cut optimization to identify individual objects within Gaussian Splatting scenes. 
Its Gradio-based interface enables users to interact with the pipeline, refine segmentation results, and export segmented objects for further editing and reuse.

### Demo Video

https://github.com/user-attachments/assets/b5df9df2-30d0-4ee4-b236-eeb4d7b09c9a

If the video does not play, [download or open it directly](media/SegmentThat.mp4), or [watch it on YouTube](https://youtu.be/cyODeUZm8M0?t=34).

### How It Works

1. **Interactive Multi-View Segmentation:** Users select informative viewpoints and specify target objects using text prompts, visual clicks, or a combination of both. [SAM 3](https://github.com/facebookresearch/sam3) generates 2D segmentation masks that users can inspect and refine before proceeding.

2. **Differentiable Foreground Attribution:** Differentiable rendering is used to estimate the contribution of individual Gaussians to the segmented foreground regions. Evidence from multiple views is aggregated to compute per-Gaussian foreground confidence scores.

3. **3D Graph-Cut Optimization:** Foreground confidence scores and relationships between neighboring Gaussians are incorporated into a graph-cut optimization to produce spatially coherent object segmentations while preserving boundaries.

### Export and Integration

Segmented objects can be exported as individual PLY files with their pivots relocated to the origin. 
The remaining scene, with all labeled objects removed, can also be exported as a single PLY file that keeps the original scene positions. 
Additionally, a binary (.bin) file stores object labels and associated metadata. 
The binary file is designed for integration with the Unity game engine, which enables objects to be identified and rendered at their original scene positions. To use this feature, you will need a modified version of the gsplat-unity plugin. It is based on the upstream [gsplat-unity](https://github.com/wuyize25/gsplat-unity) and includes additional changes for our use cases:

- **Unity Plugin:** [https://github.com/FilmakademieRnd/SegmentThat_GSplat_UnityPlugin]


## Setup

### Requirements

- Linux or [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
- NVIDIA driver with CUDA support
- [uv](https://docs.astral.sh/uv/) for Python and dependency management
- Python 3.11–3.13 (Python 3.12.3 tested); `uv` can automatically install a compatible Python version
- A CUDA-capable NVIDIA GPU is strongly recommended
- A Hugging Face account with access to the [SAM 3 model weights](https://huggingface.co/facebook/sam3)

### Install uv

If `uv` is not already installed, follow the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Installation

Download the source code or clone the GitHub repository. Run all commands from the project root directory in the terminal.

**WSL2 users:** If your project is located on the Windows filesystem, place the virtual environment on the Linux filesystem to avoid filesystem performance issues:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/segmentthat"
```

Skip this step if your project is located on the Linux filesystem.

**1. Install dependencies**

```bash
uv sync
```

This creates or updates the project environment and installs the dependencies defined in `pyproject.toml`, including PyTorch, SAM3, and gsplat.

SAM3 is configured to install directly from its [Git repository](https://github.com/facebookresearch/sam3) through the project's `pyproject.toml` configuration.

**2. Verify and Import PyTorch, CUDA, and gsplat**

```bash
uv run python -c "import torch, gsplat; print('PyTorch:', torch.__version__); print('gsplat:', gsplat.__file__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

This command verifies that PyTorch and gsplat can be imported, displays their installation information, and checks CUDA availability.

For GPU-accelerated segmentation, `CUDA available` should be `True`. Ensure your NVIDIA driver is compatible with the installed CUDA-enabled PyTorch build.

**3. Verify SAM 3 Installation**

```bash
uv run python -c "from sam3 import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor; print('SAM3 import OK')"
```

The installation is successful when the command completes and prints `SAM3 import OK`.

**Configure SAM 3 access**

Request access to the [SAM 3 model weights](https://huggingface.co/facebook/sam3). Once access is granted, authenticate with Hugging Face:

```bash
uv run hf auth login
```

**4. Launch the application**

```bash
uv run src/app.py
```

### Accessing the Application

The application uses the following local addresses:

- **Gradio:** http://127.0.0.1:7866
- **Viser 3D viewer:** http://127.0.0.1:8080

Open the Gradio address in a browser. The 3D viewer is embedded in the Gradio page. If the embedded viewer does not appear, open the Viser address directly in another browser tab and reload the Gradio page.

### Dependency Management

All project dependencies are managed by `uv` and declared in `pyproject.toml`, with exact resolved versions recorded in `uv.lock`.

- Use `uv add <package>` to add a dependency.
- Use `uv sync` to synchronize the environment with the project configuration.
- Commit both `pyproject.toml` and `uv.lock` to version control to support reproducible installations.

Avoid installing project dependencies directly with `pip` unless a package's installation instructions explicitly require an additional step that `uv` does not handle.


## License

Copyright (C) 2026 [Filmakademie Baden-Württemberg](https://www.filmakademie.de/), [Carl Laemmle Institut for Research and Media Innovation](https://cli.filmakademie.de/).

This project is licensed under the MIT License (see [LICENSE](LICENSE)).

### Third-Party Components

This project depends on third-party components (SAM 3, Gradio, Viser, gsplat, PyTorch) that are subject to their own licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
