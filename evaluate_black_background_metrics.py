#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无真值的“去水体/黑背景”评价指标。

本脚本只需要一张 RGB 渲染图，计算三个指标：

* VFBP (Veil-Free Background Purity, ↑)：边界连通背景的黑度、低色度和
  低纹理程度；
* FSI (Foreground Structure Integrity, ↑)：前景多尺度边缘稳定性、结构张量
  方向一致性和细节强度，同时抑制平坦区域噪声；
* BHS (Boundary Halo Suppression, ↑)：前景/背景边界附近的亮度、色度和
  局部纹理泄漏。BHS 在没有足够深层背景区域时返回 null，而不是猜测。

背景区域由当前图像自动估计：先计算亮度梯度和局部亮度标准差，选出低活动
像素，再只保留与图像边界连通的区域。因此不需要 Alpha、深度、相机参数、
原始水下图像或真值图像。

推荐用法（在 3D-UIR 环境中）：

    conda activate 3D-UIR
    python evaluate_black_background_metrics.py \
        --pred-dir /path/to/intrinsic_color_render

默认输出 ``<pred-dir>/black_background_metrics.json``。例如：

    python evaluate_black_background_metrics.py \
        --pred-dir /home/gyz/seafree2/outputs/AquaNull-LVSF-v8/IUI3-RedSea_undistorted/test_intrinsic/test/intrinsic_color_render \
        --output /tmp/redsea_black_metrics.json \
        --csv-output /tmp/redsea_black_metrics.csv

论文实验时请在开发集固定参数，不要根据测试集调节阈值。不同方法必须使用
相同的脚本和参数。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EPS = 1.0e-8


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return values[np.isfinite(values)]


def _quantile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    values = _finite_values(values)
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, q))


def _odd_kernel(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)


def _safe_normalize(values: np.ndarray, q95: Optional[float] = None) -> np.ndarray:
    """Normalize by a robust percentile and clip to [0, 1]."""
    if q95 is None:
        q95 = _quantile(values, 95.0, default=0.0)
    if not np.isfinite(q95) or q95 <= EPS:
        return np.zeros_like(values, dtype=np.float32)
    return np.minimum(np.asarray(values, dtype=np.float32) / (q95 + EPS), 1.0)


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def _image_features(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Return sRGB luminance, Lab chroma, local std and gradient magnitude."""
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    # BT.709 luminance. We intentionally operate in the saved sRGB domain so
    # that the metric is reproducible from the rendered image alone.
    y = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)

    rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV stores a,b in [0,255] with neutral value 128. The denominator is
    # sqrt(2*127^2), the largest possible distance from neutral in this space.
    chroma = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)
    chroma = (chroma / math.sqrt(2.0 * 127.0 * 127.0)).astype(np.float32)

    mean = cv2.GaussianBlur(y, (0, 0), sigmaX=3.0, sigmaY=3.0)
    mean_sq = cv2.GaussianBlur(y * y, (0, 0), sigmaX=3.0, sigmaY=3.0)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)

    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return {
        "y": y,
        "chroma": chroma,
        "local_std": local_std,
        "gradient": gradient,
    }


def _boundary_connected_background(
    activity: np.ndarray,
    threshold: float,
    close_radius: int,
) -> np.ndarray:
    """Keep only low-activity components touching the image boundary."""
    candidate = (activity <= float(threshold)).astype(np.uint8)
    if not np.any(candidate):
        return np.zeros_like(candidate, dtype=bool)

    count, labels, _, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if count <= 1:
        return np.zeros_like(candidate, dtype=bool)

    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.zeros_like(candidate, dtype=bool)

    background = np.isin(labels, border_labels).astype(np.uint8)
    if close_radius > 0:
        background = cv2.morphologyEx(
            background, cv2.MORPH_CLOSE, _odd_kernel(close_radius)
        )
    return background.astype(bool)


def _estimate_masks(
    features: Dict[str, np.ndarray],
    activity_threshold: float,
    close_radius: int,
    foreground_erode_radius: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gradient = features["gradient"]
    local_std = features["local_std"]
    activity = 0.5 * _safe_normalize(gradient) + 0.5 * _safe_normalize(local_std)
    background = _boundary_connected_background(
        activity, activity_threshold, close_radius
    )
    foreground_raw = ~background
    # Exclude a narrow band around the automatically estimated boundary from
    # FSI, so a halo cannot be counted as foreground detail.
    if foreground_erode_radius > 0:
        foreground_eval = cv2.erode(
            foreground_raw.astype(np.uint8),
            _odd_kernel(foreground_erode_radius),
        ).astype(bool)
    else:
        foreground_eval = foreground_raw
    return background, foreground_raw, foreground_eval


def _vfbp(
    features: Dict[str, np.ndarray], background: np.ndarray
) -> Tuple[float, bool, Dict[str, float]]:
    """Veil-Free Background Purity (higher is better)."""
    if not np.any(background):
        return 0.0, False, {"mean_y": 0.0, "q95_y": 0.0, "mean_chroma": 0.0, "q95_chroma": 0.0, "mean_std": 0.0, "q95_std": 0.0}

    y = features["y"][background]
    c = features["chroma"][background]
    s = features["local_std"][background]

    # Mean captures the amount of contamination; Q95 prevents sparse bright or
    # coloured splats from being hidden by a large black area.
    ry = 0.5 * (float(np.mean(y)) + _quantile(y, 95.0))
    rc = 0.5 * (float(np.mean(c)) + _quantile(c, 95.0))
    rs = 0.5 * (float(np.mean(s)) + _quantile(s, 95.0))
    exponent = (ry / 0.05 + rc / 0.05 + rs / 0.02) / 3.0
    value = float(100.0 * math.exp(-min(max(exponent, 0.0), 80.0)))
    details = {
        "mean_y": float(np.mean(y)),
        "q95_y": _quantile(y, 95.0),
        "mean_chroma": float(np.mean(c)),
        "q95_chroma": _quantile(c, 95.0),
        "mean_std": float(np.mean(s)),
        "q95_std": _quantile(s, 95.0),
    }
    return value, True, details


def _f1_tolerance(a: np.ndarray, b: np.ndarray, radius: int) -> float:
    """Symmetric F1 after allowing a small spatial edge displacement."""
    na = int(np.count_nonzero(a))
    nb = int(np.count_nonzero(b))
    if na == 0 or nb == 0:
        return 0.0
    kernel = _odd_kernel(radius)
    b_near = cv2.dilate(b.astype(np.uint8), kernel).astype(bool)
    a_near = cv2.dilate(a.astype(np.uint8), kernel).astype(bool)
    precision = float(np.count_nonzero(a & b_near)) / float(na)
    recall = float(np.count_nonzero(b & a_near)) / float(nb)
    return float(2.0 * precision * recall / (precision + recall + EPS))


def _fsi(
    features: Dict[str, np.ndarray],
    foreground_raw: np.ndarray,
    foreground_eval: np.ndarray,
    edge_tolerance: int,
    min_foreground_fraction: float,
    min_eval_fraction: float,
) -> Tuple[float, bool, Dict[str, float]]:
    """Foreground Structure Integrity (higher is better)."""
    raw_fraction = float(np.mean(foreground_raw))
    eval_fraction = float(np.mean(foreground_eval))
    details: Dict[str, float] = {
        "foreground_fraction": raw_fraction,
        "foreground_eval_fraction": eval_fraction,
        "detail": 0.0,
        "edge_persistence": 0.0,
        "structure_coherence": 0.0,
        "noise_sigma": 0.0,
    }
    # An all-black (or almost all-black) output must not obtain a good FSI.
    if raw_fraction < min_foreground_fraction or eval_fraction < min_eval_fraction:
        return 0.0, False, details

    y = features["y"]
    f = foreground_eval
    y_f = y[f]
    dynamic = max(_quantile(y_f, 95.0) - _quantile(y_f, 5.0), 1.0e-3)

    edge_sets: List[np.ndarray] = []
    detail_terms: List[float] = []
    coherence_terms: List[float] = []
    gradients: List[np.ndarray] = []
    for sigma in (1.0, 2.0, 4.0):
        smooth = cv2.GaussianBlur(y, (0, 0), sigmaX=sigma, sigmaY=sigma)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        gradients.append(grad)

        q75 = _quantile(grad[f], 75.0)
        detail_terms.append(float(q75 / (q75 + 0.1 * dynamic + EPS)))
        threshold = _quantile(grad[f], 80.0)
        edge = (grad > threshold) & f if threshold > EPS else np.zeros_like(f)
        edge_sets.append(edge)

        # Structure tensor coherence is evaluated only on strong edges.
        a = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=1.2, sigmaY=1.2)
        d = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=1.2, sigmaY=1.2)
        cross = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=1.2, sigmaY=1.2)
        coherence = np.sqrt((a - d) ** 2 + 4.0 * cross * cross) / (a + d + EPS)
        coherence = np.clip(coherence, 0.0, 1.0)
        coherence_terms.append(
            float(np.mean(coherence[edge])) if np.any(edge) else 0.0
        )

    edge_persistence = 0.5 * (
        _f1_tolerance(edge_sets[0], edge_sets[1], edge_tolerance)
        + _f1_tolerance(edge_sets[1], edge_sets[2], edge_tolerance)
    )
    detail = float(np.mean(detail_terms))
    # Use the middle scale for the reported coherence; averaging the three
    # scales makes the value less sensitive to a single noisy scale.
    structure_coherence = float(np.mean(coherence_terms))

    # Robust noise estimate in low-gradient foreground regions.
    g1 = gradients[0]
    flat = f & (g1 <= _quantile(g1[f], 30.0))
    residual = np.abs(y - cv2.GaussianBlur(y, (0, 0), sigmaX=1.0, sigmaY=1.0))
    noise_sigma = (
        float(1.4826 * np.median(residual[flat])) if np.any(flat) else 0.0
    )

    value = 100.0 * detail * edge_persistence * structure_coherence * math.exp(
        -noise_sigma / (2.0 / 255.0)
    )
    details.update(
        {
            "detail": detail,
            "edge_persistence": float(edge_persistence),
            "structure_coherence": structure_coherence,
            "noise_sigma": noise_sigma,
        }
    )
    return float(np.clip(value, 0.0, 100.0)), True, details


def _bhs(
    features: Dict[str, np.ndarray],
    background: np.ndarray,
    ring_radius: int,
    min_ring_pixels: int,
) -> Tuple[Optional[float], bool, Dict[str, Any]]:
    """Boundary Halo Suppression (higher is better)."""
    details: Dict[str, Any] = {
        "ring_radius_px": int(ring_radius),
        "ring_pixels": [],
        "deep_pixels": 0,
        "mean_halo_y": None,
        "mean_halo_chroma": None,
        "mean_halo_std": None,
    }
    if not np.any(background) or ring_radius <= 0:
        return None, False, details

    distance = cv2.distanceTransform(background.astype(np.uint8), cv2.DIST_L2, 3)
    annuli = [
        background & (distance > 0.0) & (distance <= ring_radius),
        background
        & (distance > ring_radius)
        & (distance <= 2 * ring_radius),
        background
        & (distance > 2 * ring_radius)
        & (distance <= 3 * ring_radius),
    ]
    deep = background & (distance > 3 * ring_radius)
    details["ring_pixels"] = [int(np.count_nonzero(a)) for a in annuli]
    details["deep_pixels"] = int(np.count_nonzero(deep))
    if (
        np.count_nonzero(deep) < min_ring_pixels
        or any(np.count_nonzero(a) < min_ring_pixels for a in annuli)
    ):
        return None, False, details

    y = features["y"]
    c = features["chroma"]
    s = features["local_std"]
    base_y = _quantile(y[deep], 50.0)
    base_c = _quantile(c[deep], 50.0)
    base_s = _quantile(s[deep], 50.0)
    terms: List[float] = []
    halo_y: List[float] = []
    halo_c: List[float] = []
    halo_s: List[float] = []
    for annulus in annuli:
        hy = max(0.0, _quantile(y[annulus], 90.0) - base_y)
        hc = max(0.0, _quantile(c[annulus], 90.0) - base_c)
        hs = max(0.0, _quantile(s[annulus], 90.0) - base_s)
        halo_y.append(hy)
        halo_c.append(hc)
        halo_s.append(hs)
        terms.append(hy / 0.05 + hc / 0.05 + hs / 0.02)

    penalty = float(np.mean(terms) / 3.0)
    value = float(100.0 * math.exp(-min(max(penalty, 0.0), 80.0)))
    details.update(
        {
            "mean_halo_y": float(np.mean(halo_y)),
            "mean_halo_chroma": float(np.mean(halo_c)),
            "mean_halo_std": float(np.mean(halo_s)),
            "deep_median_y": base_y,
            "deep_median_chroma": base_c,
            "deep_median_std": base_s,
        }
    )
    return value, True, details


def evaluate_image(
    path: Path,
    *,
    activity_threshold: float,
    mask_close_radius: int,
    foreground_erode_radius_fraction: float,
    edge_tolerance_px: int,
    min_foreground_fraction: float,
    min_eval_fraction: float,
    halo_radius_fraction: float,
    min_ring_pixels: int,
) -> Dict[str, Any]:
    rgb = _load_rgb(path)
    h, w = rgb.shape[:2]
    min_dim = min(h, w)
    features = _image_features(rgb)
    erode_radius = max(0, int(round(foreground_erode_radius_fraction * min_dim)))
    halo_radius = max(1, int(round(halo_radius_fraction * min_dim)))
    background, foreground_raw, foreground_eval = _estimate_masks(
        features,
        activity_threshold=activity_threshold,
        close_radius=mask_close_radius,
        foreground_erode_radius=erode_radius,
    )

    vfbp, vfbp_valid, vfbp_details = _vfbp(features, background)
    fsi, fsi_valid, fsi_details = _fsi(
        features,
        foreground_raw,
        foreground_eval,
        edge_tolerance=edge_tolerance_px,
        min_foreground_fraction=min_foreground_fraction,
        min_eval_fraction=min_eval_fraction,
    )
    bhs, bhs_valid, bhs_details = _bhs(
        features,
        background,
        ring_radius=halo_radius,
        min_ring_pixels=min_ring_pixels,
    )

    warnings: List[str] = []
    if not vfbp_valid:
        warnings.append("未检测到与图像边界连通的低活动背景区域")
    if not fsi_valid:
        warnings.append("前景面积过小或腐蚀后无足够像素，FSI 记为 0（退化输出）")
    if not bhs_valid:
        warnings.append("背景区域没有足够的三层边界环和深层区域，BHS 为 null")

    return {
        "vfbp": float(vfbp),
        "vfbp_valid": bool(vfbp_valid),
        "fsi": float(fsi),
        "fsi_valid": bool(fsi_valid),
        "bhs": None if bhs is None else float(bhs),
        "bhs_valid": bool(bhs_valid),
        "background_fraction": float(np.mean(background)),
        "foreground_fraction": float(np.mean(foreground_raw)),
        "foreground_eval_fraction": float(np.mean(foreground_eval)),
        "height": int(h),
        "width": int(w),
        "warnings": warnings,
        "vfbp_details": vfbp_details,
        "fsi_details": fsi_details,
        "bhs_details": bhs_details,
    }


def _list_images(pred_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    iterator: Iterable[Path] = pred_dir.rglob(pattern) if recursive else pred_dir.glob(pattern)
    return sorted(
        [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: str(p.relative_to(pred_dir)),
    )


def _aggregate(per_image: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"mean": {}, "std": {}, "valid_count": {}}
    for key in ("vfbp", "fsi", "bhs"):
        values = [
            float(item[key])
            for item in per_image.values()
            if item.get(key) is not None and np.isfinite(float(item[key]))
        ]
        result["valid_count"][key] = len(values)
        if values:
            result["mean"][key] = float(np.mean(values))
            result["std"][key] = float(np.std(values))
        else:
            result["mean"][key] = None
            result["std"][key] = None
    return result


def _write_csv(path: Path, per_image: Dict[str, Dict[str, Any]]) -> None:
    fields = [
        "image",
        "vfbp",
        "fsi",
        "bhs",
        "vfbp_valid",
        "fsi_valid",
        "bhs_valid",
        "background_fraction",
        "foreground_fraction",
        "foreground_eval_fraction",
        "width",
        "height",
        "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, item in per_image.items():
            row = {field: item.get(field) for field in fields if field != "image"}
            row["image"] = name
            row["warnings"] = " | ".join(item.get("warnings", []))
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute VFBP, FSI and BHS without a reference image."
    )
    parser.add_argument("--pred-dir", type=Path, required=True, help="渲染图像目录")
    parser.add_argument(
        "--glob",
        default="*",
        help="文件匹配模式，默认 *；例如 '*.jpg' 或 'eval_*.png'",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="递归搜索 pred-dir 的子目录"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON 输出路径；默认 <pred-dir>/black_background_metrics.json",
    )
    parser.add_argument("--csv-output", type=Path, default=None, help="可选 CSV 输出路径")
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=0.20,
        help="背景低活动阈值（论文默认 0.20）",
    )
    parser.add_argument(
        "--mask-close-radius",
        type=int,
        default=1,
        help="背景掩膜形态学闭运算半径（像素，默认 1）",
    )
    parser.add_argument(
        "--foreground-erode-fraction",
        type=float,
        default=0.005,
        help="FSI 前景腐蚀半径占短边比例（默认 0.005）",
    )
    parser.add_argument(
        "--edge-tolerance-px",
        type=int,
        default=3,
        help="多尺度边缘 F1 的空间容差（默认 3 像素）",
    )
    parser.add_argument(
        "--min-foreground-fraction",
        type=float,
        default=0.05,
        help="FSI 退化检测的最小原始前景比例（默认 0.05）",
    )
    parser.add_argument(
        "--min-eval-fraction",
        type=float,
        default=0.01,
        help="FSI 退化检测的最小腐蚀后前景比例（默认 0.01）",
    )
    parser.add_argument(
        "--halo-radius-fraction",
        type=float,
        default=0.01,
        help="BHS 每层边界环宽度占短边比例（默认 0.01）",
    )
    parser.add_argument(
        "--min-ring-pixels",
        type=int,
        default=32,
        help="BHS 每层环和深层背景的最少像素数（默认 32）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = args.pred_dir.expanduser().resolve()
    if not pred_dir.is_dir():
        raise SystemExit(f"pred-dir 不存在或不是目录: {pred_dir}")

    paths = _list_images(pred_dir, args.glob, args.recursive)
    if not paths:
        raise SystemExit(f"目录中没有匹配的图像: {pred_dir} / {args.glob}")

    per_image: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        relative_name = str(path.relative_to(pred_dir))
        try:
            per_image[relative_name] = evaluate_image(
                path,
                activity_threshold=args.activity_threshold,
                mask_close_radius=args.mask_close_radius,
                foreground_erode_radius_fraction=args.foreground_erode_fraction,
                edge_tolerance_px=args.edge_tolerance_px,
                min_foreground_fraction=args.min_foreground_fraction,
                min_eval_fraction=args.min_eval_fraction,
                halo_radius_fraction=args.halo_radius_fraction,
                min_ring_pixels=args.min_ring_pixels,
            )
        except Exception as exc:  # keep a broken file from hiding other results
            per_image[relative_name] = {
                "vfbp": None,
                "vfbp_valid": False,
                "fsi": None,
                "fsi_valid": False,
                "bhs": None,
                "bhs_valid": False,
                "warnings": [f"评估失败: {type(exc).__name__}: {exc}"],
            }

    aggregate = _aggregate(per_image)
    output = args.output or (pred_dir / "black_background_metrics.json")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "metric_name": {
            "vfbp": "Veil-Free Background Purity (higher is better)",
            "fsi": "Foreground Structure Integrity (higher is better)",
            "bhs": "Boundary Halo Suppression (higher is better)",
        },
        "description": "No-reference metrics computed from rendered RGB images only.",
        "pred_dir": str(pred_dir),
        "glob": args.glob,
        "count": len(per_image),
        "config": {
            "activity_threshold": args.activity_threshold,
            "mask_close_radius": args.mask_close_radius,
            "foreground_erode_fraction": args.foreground_erode_fraction,
            "edge_tolerance_px": args.edge_tolerance_px,
            "min_foreground_fraction": args.min_foreground_fraction,
            "min_eval_fraction": args.min_eval_fraction,
            "halo_radius_fraction": args.halo_radius_fraction,
            "min_ring_pixels": args.min_ring_pixels,
        },
        "mean": aggregate["mean"],
        "std": aggregate["std"],
        "valid_count": aggregate["valid_count"],
        "per_image": per_image,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)

    if args.csv_output is not None:
        csv_output = args.csv_output.expanduser().resolve()
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(csv_output, per_image)

    print(f"Evaluated {len(per_image)} image(s) from {pred_dir}")
    print(f"VFBP: mean={aggregate['mean']['vfbp']} std={aggregate['std']['vfbp']}")
    print(f"FSI : mean={aggregate['mean']['fsi']} std={aggregate['std']['fsi']}")
    print(
        f"BHS : mean={aggregate['mean']['bhs']} std={aggregate['std']['bhs']} "
        f"(valid={aggregate['valid_count']['bhs']}/{len(per_image)})"
    )
    print(f"JSON: {output}")
    if args.csv_output is not None:
        print(f"CSV : {args.csv_output.expanduser().resolve()}")


if __name__ == "__main__":
    main()

