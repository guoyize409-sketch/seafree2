# gsplat for SeaFree-GS

This directory contains the local `gsplat 1.4.0` dependency used by
SeaFree-GS.

SeaFree-GS keeps the upstream rasterization implementation and uses a small
project-specific change in:

- `gsplat/strategy/default.py`

The local strategy change is limited to the Gaussian refinement schedule:

- configurable opacity reset through `reset_alpha_value`
- optional low-opacity pruning after densification stops through
  `continue_cull_post_densification`
- a separate post-densification opacity pruning threshold through
  `cull_alpha_thresh_post`

This refinement adjustment was inspired by prior underwater and
opacity-aware Gaussian splatting methods, including WaterSplatting, while the
implementation here is kept intentionally minimal for SeaFree-GS.

These changes are used by the SeaFree-GS training configuration to keep the
Gaussian set clean after densification has stopped, while preserving the
default gsplat gradient accumulation and split/duplicate behavior.

## Installation

Install this local version from source:

```bash
pip uninstall -y gsplat
cd third_party/gsplat
pip install -e .
```

Please avoid mixing this local copy with another pip-installed version of
`gsplat` in the same environment. SeaFree-GS expects this bundled version when
using its default training configuration.

## Upstream Project

Upstream repository:

- [https://github.com/nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)

We gratefully acknowledge the authors and contributors of the original gsplat
project.

## Original Citation

If you find the original gsplat library useful in your projects or papers,
please consider citing:

```bibtex
@article{ye2024gsplatopensourcelibrarygaussian,
    title={gsplat: An Open-Source Library for {Gaussian} Splatting},
    author={Vickie Ye and Ruilong Li and Justin Kerr and Matias Turkulainen and Brent Yi and Zhuoyang Pan and Otto Seiskari and Jianbo Ye and Jeffrey Hu and Matthew Tancik and Angjoo Kanazawa},
    year={2024},
    eprint={2409.06765},
    journal={arXiv preprint arXiv:2409.06765},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2409.06765}
}
```

## License

This directory is derived from the original gsplat project. Please refer to the
original license files included in this directory for licensing details.
