#!/usr/bin/env python3
"""Evaluate no-reference underwater image quality metrics.

This entry point follows the public evaluation code used by FUnIE-GAN for
UIQM (``measure_uiqm.py`` + ``uqim_utils.py``), the official UCIQE reference
implementation (JOU-UIP/UCIQE), and scikit-video's NIQE implementation.  It
does not compare against a clear/reference image, so ``--gt-dir`` is not
needed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_uiqm():
    """Load the public FUnIE-GAN UIQM implementation.

    The upstream file predates Python 3 and uses float slice bounds.  The
    original equations are kept intact; only the two block-count values are
    converted to integers so the published implementation runs on Python 3.
    """
    here = Path(__file__).resolve().parent
    vendored = here / "uqim_utils.py"
    if not vendored.exists():
        raise FileNotFoundError(
            f"Missing {vendored}. Download the public FUnIE-GAN file "
            "Evaluation/uqim_utils.py into the 3d-uir directory."
        )
    namespace: dict[str, object] = {}
    source = vendored.read_text(encoding="utf-8")
    source = source.replace("k1 = x.shape[1]/window_size", "k1 = int(x.shape[1]/window_size)")
    source = source.replace("k2 = x.shape[0]/window_size", "k2 = int(x.shape[0]/window_size)")
    # FUnIE-GAN's public evaluator resizes every image to 256x256 before
    # calling getUIQM; callers can reproduce that with --uiqm-resize.
    exec(compile(source, str(vendored), "exec"), namespace)
    return namespace["getUIQM"]


def _uciqe(path: Path) -> float:
    """Call the public JOU-UIP UCIQE.py implementation."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from UCIQE import uciqe
    return float(uciqe(1, str(path)))


def _niqe_fn():
    # sk-video ships the original NIQE model parameters with the 3D-UIR env.
    try:
        import skvideo.measure  # noqa: F401
        import skvideo.measure.niqe as niqe_module
    except ImportError as exc:
        raise RuntimeError("scikit-video is required for NIQE in the 3D-UIR environment") from exc
    # sk-video 1.1.10 still calls np.int and scipy.misc.imresize.  Patch these
    # compatibility points without changing its NIQE equations or parameters.
    np.int = int  # type: ignore[attr-defined]
    import scipy.misc
    if not hasattr(scipy.misc, "imresize"):
        def imresize(arr, scale, interp="bicubic", mode="F"):
            arr = np.asarray(arr, dtype=np.float32)
            h, w = arr.shape[:2]
            resized = Image.fromarray(arr, mode="F").resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.Resampling.BICUBIC,
            )
            return np.asarray(resized, dtype=np.float32)
        scipy.misc.imresize = imresize  # type: ignore[attr-defined]
    return skvideo.measure.niqe


def _niqe(path: Path, niqe) -> float:
    # NIQE is defined on luminance. cv2 grayscale uses the standard BT.601
    # luminance transform, matching common underwater benchmark scripts.
    bgr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if bgr is None:
        raise ValueError(f"Unable to read image: {path}")
    score = niqe(bgr.astype(np.float32)[None, :, :, None])
    return float(np.asarray(score).reshape(-1)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="UIQM/UCIQE/NIQE no-reference evaluation")
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--pred-glob", default="*", help="Glob under pred-dir (default: all image files)")
    parser.add_argument("--uiqm-resize", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=(256, 256),
                        help="Resize used by the public FUnIE-GAN UIQM script (default: 256 256)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pred_dir = args.pred_dir.resolve()
    if not pred_dir.is_dir():
        print(f"[ERROR] prediction directory does not exist: {pred_dir}", file=sys.stderr)
        return 1
    files = sorted((p for p in pred_dir.glob(args.pred_glob) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                   key=lambda p: p.name)
    if not files:
        print(f"[ERROR] no images found in {pred_dir} matching {args.pred_glob}", file=sys.stderr)
        return 1

    get_uiqm = _load_uiqm()
    niqe = _niqe_fn()
    per_image = {}
    uiqm_values, uciqe_values, niqe_values = [], [], []
    width, height = args.uiqm_resize
    for path in files:
        try:
            rgb = np.asarray(Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BICUBIC))
            uiqm = float(get_uiqm(rgb))
            uciqe = float(_uciqe(path))
            niqe_value = float(_niqe(path, niqe))
            values = {"uiqm": uiqm, "uciqe": uciqe, "niqe": niqe_value}
            uiqm_values.append(uiqm); uciqe_values.append(uciqe); niqe_values.append(niqe_value)
        except Exception as exc:
            values = {"error": str(exc)}
        per_image[path.name] = values
        print(f"{path.name}: UIQM={values.get('uiqm', float('nan')):.9f} "
              f"UCIQE={values.get('uciqe', float('nan')):.9f} NIQE={values.get('niqe', float('nan')):.9f}")

    if not uiqm_values:
        print("[ERROR] no image could be evaluated", file=sys.stderr)
        return 1
    mean = {"uiqm": float(np.mean(uiqm_values)), "uciqe": float(np.mean(uciqe_values)), "niqe": float(np.mean(niqe_values))}
    std = {"uiqm": float(np.std(uiqm_values)), "uciqe": float(np.std(uciqe_values)), "niqe": float(np.std(niqe_values))}
    results = {
        "pred_dir": str(pred_dir),
        "count": len(files),
        "evaluated_count": len(uiqm_values),
        "uiqm_resize": [width, height],
        "implementation": {
            "uiqm": "FUnIE-GAN Evaluation/measure_uiqm.py + uqim_utils.py",
            "uciqe": "JOU-UIP/UCIQE UCIQE.py",
            "niqe": "scikit-video skvideo.measure.niqe (bundled NIQE model parameters)",
        },
        "mean": mean,
        "std": std,
        "per_image": per_image,
    }
    output = (args.output or pred_dir / "underwater_metrics_results.json").resolve()
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Mean: UIQM={mean['uiqm']:.9f} UCIQE={mean['uciqe']:.9f} NIQE={mean['niqe']:.9f}")
    print(f"Std:  UIQM={std['uiqm']:.9f} UCIQE={std['uciqe']:.9f} NIQE={std['niqe']:.9f}")
    print(f"Results written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
