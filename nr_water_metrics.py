#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-reference water-medium removal metrics for dewatered underwater renders.

Metrics
-------
1) SCRS ↑ : Structure-Conditioned Radiance Suppression
2) SCCS ↑ : Structure-Conditioned Chromatic Suppression
3) RSC  ↑ : Radiance-Structure Coupling (Spearman correlation)

Only the final rendered RGB image is required. No GT, depth, mask, original
underwater image, camera parameters, or 3D representation is used.

Recommended fixed protocol for paper evaluation:
    q = 0.30
    sigma_grad = 1.0
    sigma_var = 3.0

Do NOT tune these parameters separately for different methods.

Dependencies:
    pip install numpy opencv-python scipy

Examples:
    python nr_water_metrics.py --input image.png
    python nr_water_metrics.py --input ./renders
    python nr_water_metrics.py --input ./renders --recursive --csv metrics.csv
    python nr_water_metrics.py --input image.png --save-debug ./debug
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError as e:
    raise ImportError(
        "scipy is required for RSC. Install dependencies with:\n"
        "    pip install numpy opencv-python scipy"
    ) from e


EPS = 1e-12
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def read_rgb(path: Path) -> np.ndarray:
    """Read image as float32 RGB in [0, 1]. Alpha, if present, is discarded."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported channel count {img.shape[2]}: {path}")
    else:
        raise ValueError(f"Unsupported image shape {img.shape}: {path}")

    if img.dtype == np.uint8:
        rgb = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        rgb = img.astype(np.float32) / 65535.0
    else:
        rgb = img.astype(np.float32)
        if np.nanmax(rgb) > 1.5:
            rgb = rgb / 255.0

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(rgb, 0.0, 1.0)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB [0,1]."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def compute_structure_map(
    rgb: np.ndarray,
    sigma_grad: float = 1.0,
    sigma_var: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute structure activity A(p):
        A = 0.5 * G/P95(G) + 0.5 * V/P95(V)

    G: gradient magnitude of Gaussian-smoothed linear luminance.
    V: local standard deviation of linear luminance.
    """
    linear = srgb_to_linear(rgb)
    y = (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    ).astype(np.float32)

    if sigma_grad > 0:
        y_g = cv2.GaussianBlur(
            y, (0, 0), sigmaX=sigma_grad, sigmaY=sigma_grad,
            borderType=cv2.BORDER_REFLECT101
        )
    else:
        y_g = y

    gx = cv2.Sobel(y_g, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Sobel(y_g, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
    grad = np.sqrt(gx * gx + gy * gy)

    if sigma_var > 0:
        mean = cv2.GaussianBlur(
            y, (0, 0), sigmaX=sigma_var, sigmaY=sigma_var,
            borderType=cv2.BORDER_REFLECT101
        )
        mean2 = cv2.GaussianBlur(
            y * y, (0, 0), sigmaX=sigma_var, sigmaY=sigma_var,
            borderType=cv2.BORDER_REFLECT101
        )
    else:
        mean = y
        mean2 = y * y

    var = np.maximum(mean2 - mean * mean, 0.0)
    local_std = np.sqrt(var)

    p95_g = float(np.percentile(grad, 95.0))
    p95_v = float(np.percentile(local_std, 95.0))

    g_hat = grad / (p95_g + EPS)
    v_hat = local_std / (p95_v + EPS)

    activity = 0.5 * g_hat + 0.5 * v_hat
    activity = np.nan_to_num(activity, nan=0.0, posinf=0.0, neginf=0.0)

    return activity.astype(np.float32), linear


def linear_radiance_magnitude(linear_rgb: np.ndarray) -> np.ndarray:
    """r(p) = sqrt((R_lin^2 + G_lin^2 + B_lin^2)/3)."""
    return np.sqrt(np.mean(linear_rgb * linear_rgb, axis=2)).astype(np.float32)


def lab_chroma(rgb: np.ndarray) -> np.ndarray:
    """CIELAB chroma C*(p) = sqrt(a*^2 + b*^2)."""
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
    a = lab[..., 1]
    b = lab[..., 2]
    return np.sqrt(a * a + b * b).astype(np.float32)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho with deterministic handling of degenerate constant arrays."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 2:
        return 0.0
    if np.ptp(x) <= EPS or np.ptp(y) <= EPS:
        return 0.0

    rho = spearmanr(x, y).statistic
    if not np.isfinite(rho):
        return 0.0
    return float(np.clip(rho, -1.0, 1.0))


def evaluate_image(
    path: Path,
    q: float = 0.30,
    sigma_grad: float = 1.0,
    sigma_var: float = 3.0,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate one rendered image. Higher is better for SCRS/SCCS/RSC."""
    if not (0.0 < q < 0.5):
        raise ValueError(f"q must satisfy 0 < q < 0.5, got {q}")

    rgb = read_rgb(path)
    activity, linear = compute_structure_map(rgb, sigma_grad, sigma_var)
    r = linear_radiance_magnitude(linear)
    c = lab_chroma(rgb)

    low_thr = float(np.quantile(activity, q))
    high_thr = float(np.quantile(activity, 1.0 - q))

    low_mask = activity <= low_thr
    high_mask = activity >= high_thr

    if not np.any(low_mask):
        low_mask = activity == np.min(activity)
    if not np.any(high_mask):
        high_mask = activity == np.max(activity)

    mu_l_r = float(np.mean(r[low_mask]))
    mu_h_r = float(np.mean(r[high_mask]))
    mu_l_c = float(np.mean(c[low_mask]))
    mu_h_c = float(np.mean(c[high_mask]))

    den_r = mu_h_r + mu_l_r
    den_c = mu_h_c + mu_l_c

    # Anti-degeneracy: fully black image => 0, not undefined/high score.
    scrs = 0.0 if den_r <= EPS else (mu_h_r - mu_l_r) / (den_r + EPS)
    sccs = 0.0 if den_c <= EPS else (mu_h_c - mu_l_c) / (den_c + EPS)
    rsc = safe_spearman(activity, r)

    metrics = {
        "SCRS": float(np.clip(scrs, -1.0, 1.0)),
        "SCCS": float(np.clip(sccs, -1.0, 1.0)),
        "RSC": float(np.clip(rsc, -1.0, 1.0)),
        "mean_r_low": mu_l_r,
        "mean_r_high": mu_h_r,
        "mean_chroma_low": mu_l_c,
        "mean_chroma_high": mu_h_c,
        "low_fraction": float(np.mean(low_mask)),
        "high_fraction": float(np.mean(high_mask)),
    }

    debug = {
        "rgb": rgb,
        "activity": activity,
        "radiance": r,
        "chroma": c,
        "low_mask": low_mask,
        "high_mask": high_mask,
    }
    return metrics, debug


def normalize_to_u8(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if hi <= lo + EPS:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_debug_images(path: Path, debug: Dict[str, np.ndarray], out_dir: Path) -> None:
    """Save structure/mask diagnostics; these do not affect metric values."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    rgb_u8 = (np.clip(debug["rgb"], 0, 1) * 255.0).astype(np.uint8)
    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

    activity_u8 = normalize_to_u8(debug["activity"])
    activity_color = cv2.applyColorMap(activity_u8, cv2.COLORMAP_TURBO)

    low = debug["low_mask"]
    high = debug["high_mask"]

    overlay = bgr_u8.copy()
    low_color = np.array([255, 0, 0], dtype=np.uint8)   # BGR blue
    high_color = np.array([0, 0, 255], dtype=np.uint8) # BGR red
    alpha = 0.45

    overlay[low] = (
        (1.0 - alpha) * overlay[low].astype(np.float32)
        + alpha * low_color.astype(np.float32)
    ).astype(np.uint8)
    overlay[high] = (
        (1.0 - alpha) * overlay[high].astype(np.float32)
        + alpha * high_color.astype(np.float32)
    ).astype(np.uint8)

    cv2.imwrite(str(out_dir / f"{stem}_structure.png"), activity_color)
    cv2.imwrite(str(out_dir / f"{stem}_low_high_overlay.png"), overlay)
    cv2.imwrite(str(out_dir / f"{stem}_low_mask.png"), low.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / f"{stem}_high_mask.png"), high.astype(np.uint8) * 255)


def collect_images(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise ValueError(f"No supported images found in: {input_path}")
    return images


def write_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image", "SCRS", "SCCS", "RSC",
        "mean_r_low", "mean_r_high",
        "mean_chroma_low", "mean_chroma_high",
        "low_fraction", "high_fraction",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_dataset_summary(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    print("\nDataset summary (mean ± std):")
    for key in ("SCRS", "SCCS", "RSC"):
        values = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        print(f"  {key:4s} ↑ : {mean:.6f} ± {std:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-reference water-medium removal metrics: SCRS, SCCS, RSC."
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Rendered image or directory containing rendered images."
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively scan subdirectories when --input is a directory."
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Optional CSV output path for per-image results."
    )
    parser.add_argument(
        "--save-debug", type=Path, default=None,
        help="Optional directory to save structure maps and low/high-structure masks."
    )
    parser.add_argument(
        "--q", type=float, default=0.30,
        help="Low/high structure quantile. Recommended fixed value: 0.30."
    )
    parser.add_argument(
        "--sigma-grad", type=float, default=1.0,
        help="Gaussian sigma before gradient computation. Default: 1.0."
    )
    parser.add_argument(
        "--sigma-var", type=float, default=3.0,
        help="Gaussian sigma for local variance. Default: 3.0."
    )

    args = parser.parse_args()

    images = collect_images(args.input, args.recursive)
    rows: List[Dict[str, object]] = []

    print(
        f"Protocol: q={args.q:.2f}, "
        f"sigma_grad={args.sigma_grad:.2f}, sigma_var={args.sigma_var:.2f}\n"
    )

    for path in images:
        try:
            metrics, debug = evaluate_image(
                path,
                q=args.q,
                sigma_grad=args.sigma_grad,
                sigma_var=args.sigma_var,
            )
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            continue

        row: Dict[str, object] = {"image": str(path)}
        row.update(metrics)
        rows.append(row)

        print(
            f"{path.name:40s} "
            f"SCRS={metrics['SCRS']:+.6f}  "
            f"SCCS={metrics['SCCS']:+.6f}  "
            f"RSC={metrics['RSC']:+.6f}"
        )

        if args.save_debug is not None:
            save_debug_images(path, debug, args.save_debug)

    if not rows:
        raise RuntimeError("No image was successfully evaluated.")

    print_dataset_summary(rows)

    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nCSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
