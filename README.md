# 🌊 SeaFree-GS: Reconstructing Underwater 3D Scenes with True Appearances

SeaFree-GS is the public training and rendering implementation for
reconstructing underwater 3D scenes with their true appearances from degraded
underwater images. It builds on 3D Gaussian Splatting and uses a
Degradation-Aware Dual-Color Modeling strategy to decouple intrinsic scene
appearance from viewpoint-dependent underwater degradation.

## 🧭 Method Overview

SeaFree-GS contains three main components:

- **Degradation-Aware Dual-Color Modeling** assigns each Gaussian an intrinsic
  color for the true scene appearance and derives its viewpoint-dependent
  degraded color through a physics-based color degradation equation.
- **Water Properties Predictor** estimates line-of-sight dependent water
  properties, including ambient light, attenuation coefficients, and
  backscatter coefficients.
- **Content-Based Loss and Coarse-Grained Depth Loss** enhance supervision over
  attenuation-affected foreground regions, supervise water-property
  optimization using background water-only regions, and introduce pseudo-depth
  constraints for coarse scene geometry.

## 📁 Repository Structure

```text
seafree_gs/
  seafree_config.py       # Nerfstudio method registration
  seafree_model.py        # SeaFree-GS model
  seafree_dataparser.py   # SeaFree-specific COLMAP dataparser
  seafree_datamanager.py  # Full-image datamanager with depth caching support

third_party/gsplat/       # Bundled local gsplat dependency
run_seafree_batch.sh      # Batch training and rendering script
```

Local datasets, training outputs, and rendered results are intentionally kept
outside version control:

```text
data/
outputs/
render_results/
```

## ⚙️ Installation

The current public release of SeaFree-GS is developed and tested on **Linux**
with the **Nerfstudio 1.1.5** ecosystem.

### 1. Create a conda environment

Nerfstudio requires **Python >= 3.8** and officially recommends using **conda**
to manage dependencies. We use a dedicated environment for SeaFree-GS:

```bash
conda create --name seafree-gs -y python=3.8
conda activate seafree-gs
python -m pip install --upgrade pip
```

### 2. Install the core dependencies

For PyTorch, CUDA, tiny-cuda-nn, and Nerfstudio, please refer to the
[official Nerfstudio installation guide](https://docs.nerf.studio/quickstart/installation.html).

One working setup for the current public release is:

```bash
pip uninstall -y torch torchvision functorch tinycudann
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install setuptools==69.5.1 packaging
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio==1.1.5
ns-install-cli
```

### 3. Replace the automatically installed gsplat

Nerfstudio may automatically install an upstream version of `gsplat`. Replace
it with the modified local version provided in:

```text
third_party/gsplat
```

> **Important:** SeaFree-GS does **not** directly use the upstream
> pip-installed `gsplat` package.

We recommend uninstalling the upstream version first, and then installing the
modified local version:

```bash
pip uninstall -y gsplat
cd third_party/gsplat
pip install -e .
cd ../../
```

See [third_party/gsplat/README.md](third_party/gsplat/README.md) for the
project-specific gsplat notes.

### 4. Install SeaFree-GS

Finally, install SeaFree-GS itself from the repository root:

```bash
pip install -e .
```

## 🗂️ Dataset Preparation

SeaFree-GS follows a COLMAP / Nerfstudio-style dataset organization.

A typical scene directory used by the current public release is:

```text
<DATA_ROOT>/
  images_wb/
  depthAnything_u16/
  sparse/
    0/
```

In [run_seafree_batch.sh](run_seafree_batch.sh), each entry in `scene_list` is
interpreted as a relative path under `data/`.

### Notes on depth input

- The current public release assumes that depth supervision is provided.
- The expected depth map format is **16-bit grayscale**.
- The expected depth semantics are **relative disparity-like depth**: near
  large, far small.
- The default example scripts use `depthAnything_u16/`, which stores results
  from DepthAnything. Other monocular depth sources can
  also be used if they follow the same convention.
- Depth filenames are expected to match the corresponding filenames in
  `images_wb/`.


## 🚀 Training

Train a single scene with:

```bash
ns-train seafree-gs --vis tensorboard \
  --data data/SeaThru/D3 \
  colmap \
  --colmap-path sparse/0 \
  --images-path images_wb \
  --depths-path depthAnything_u16
```

The default SeaFree-GS method configuration already enables the release
settings used by this package, including antialiased rasterization,
SH-degree-0 color optimization, the SeaFree losses, and the opacity reset /
post-densification pruning settings for the bundled gsplat strategy.

## 🧪 Batch Script

For batch training and rendering, edit `scene_list` in
[run_seafree_batch.sh](run_seafree_batch.sh), then run:

```bash
bash run_seafree_batch.sh
```

The script writes training logs and checkpoints to `outputs/`, and renders the
main visual results to `render_results/`.

## 🎥 Rendering Outputs

The batch script renders the following outputs by default:

- `rgb`: rendered degraded underwater image
- `intrinsic_color_render`: rendered intrinsic true-appearance image
- `depth`: rendered depth map

Example:

```bash
ns-render dataset \
  --load-config outputs/<experiment>/seafree-gs/<run>/config.yml \
  --rendered-output-names intrinsic_color_render \
  --output-path render_results/<experiment>
```

## 📚 Citation

If you use SeaFree-GS, please cite our work:

```bibtex
@ARTICLE{10989758,
  author={Liu, Shaohua and Gao, Ning and Fu, Shaowen and Zhong, Xiaoqing and Li, Hongjue},
  journal={IEEE Signal Processing Letters},
  title={SeaFree-GS: Reconstructing Underwater 3D Scenes With True Appearances},
  year={2025},
  volume={32},
  number={},
  pages={2114-2118},
  doi={10.1109/LSP.2025.3567853}
}
```

## 🙏 Acknowledgements

SeaFree-GS builds on the Nerfstudio ecosystem, the original 3D Gaussian
Splatting work, and the gsplat rasterization library. The bundled gsplat
strategy refinement is inspired by prior underwater and opacity-aware Gaussian
splatting methods, including WaterSplatting. The pseudo-depth supervision uses
monocular depth estimates such as DepthAnything.

## 📄 License

SeaFree-GS is released under the MIT License. The bundled `third_party/gsplat`
directory is derived from the original gsplat project; see the license files
included in that directory for details.
