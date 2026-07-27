# metrics.py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np
import re

# EXR single-channel reader (depth). Prefer OpenEXR when available, but allow
# the seafree-gs env to evaluate with OpenCV-only EXR support.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
try:
    import OpenEXR
    import Imath
except Exception:
    OpenEXR = None
    Imath = None
try:
    import cv2
except Exception:
    cv2 = None

def readImages(renders_dir: Path, gt_dir: Path):
    
    render_names = sorted([f for f in os.listdir(renders_dir) if not f.startswith('.')])
    gt_names     = sorted([f for f in os.listdir(gt_dir) if not f.startswith('.')])
    names = sorted(list(set(render_names).intersection(set(gt_names))))
    renders, gts, image_names = [], [], []
    for fname in names:
        render = Image.open(renders_dir / fname)
        gt     = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def read_exr_channel(file_path: str, channel_name: str = "R") -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if OpenEXR is None:
        if cv2 is None:
            raise RuntimeError("OpenEXR is unavailable and cv2 fallback is not installed.")
        arr = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"Failed to read EXR with cv2: {file_path}")
        if arr.ndim == 3:
            requested = (channel_name or "").lower()
            channel_map = {"b": 0, "g": 1, "r": 2, "z": 0, "depth": 0}
            arr = arr[..., channel_map.get(requested, 0)]
        return np.asarray(arr, dtype=np.float32)
    exr_file = OpenEXR.InputFile(file_path)
    try:
        header = exr_file.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        channels = list(header["channels"].keys())
        ch = channel_name if channel_name in channels else None
        if ch is None:
            lower_map = {c.lower(): c for c in channels}
            # allow automatic/common fallbacks if requested channel is absent
            requested = (channel_name or "").lower()
            if requested in lower_map:
                ch = lower_map[requested]
            else:
                # suffix match like A.R or something.R
                cand = [c for c in channels if c.split(".")[-1].lower() == requested and requested != "auto"]
                if len(cand) == 1:
                    ch = cand[0]
            if ch is None:
                # auto-pick common depth channels
                prefer = ["z", "depth", "r", "y"]
                for pref in prefer:
                    if pref in lower_map:
                        ch = lower_map[pref]
                        break
            if ch is None and len(channels) == 1:
                # single channel EXR
                ch = channels[0]
            if ch is None and channels:
                # last resort: first channel
                ch = channels[0]
        if ch is None:
            raise KeyError(f"EXR missing channel {channel_name}. Available: {channels}")
        try:
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            buf = exr_file.channel(ch, pt)
            arr = np.frombuffer(buf, dtype=np.float32)
        except Exception:
            pt = Imath.PixelType(Imath.PixelType.HALF)
            buf = exr_file.channel(ch, pt)
            arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        arr = arr.reshape((height, width))
        return arr
    finally:
        del exr_file


def _find_depth_pred_for_name(depth_dir: Path, stem: str) -> Path:
    """Find a predicted depth EXR in depth_dir matching image stem, robust to zero-padding and prefixes.

    Tries:
      - {stem}.exr, depth_{stem}.exr
      - Numeric-equal names with variable zero padding (e.g., 00000 -> depth_0000.exr)
      - Any *.exr whose trailing number equals the stem's integer value
    """
    # direct attempts
    direct = [depth_dir / f"{stem}.exr", depth_dir / f"depth_{stem}.exr"]
    for p in direct:
        if p.exists():
            return p

    # if stem is purely numeric, try variable padding
    num_match = re.fullmatch(r"\d+", stem)
    if num_match:
        idx = int(stem)
        # try a reasonable range of paddings
        for pad in range(1, max(2, len(stem) + 1)):
            s = f"{idx:0{pad}d}"
            cands = [depth_dir / f"{s}.exr", depth_dir / f"depth_{s}.exr"]
            for p in cands:
                if p.exists():
                    return p

        # final fallback: scan all exr and match by numeric value at end
        try:
            for name in sorted(os.listdir(depth_dir)):
                if not name.lower().endswith('.exr'):
                    continue
                m = re.search(r"(\d+)\.exr$", name)
                if m and int(m.group(1)) == idx:
                    cand = depth_dir / name
                    if cand.exists():
                        return cand
        except FileNotFoundError:
            pass

    return None

def _dir_has_exr(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.exr"))

def _resolve_gt_depth_dir(source_path: str) -> Path:
    if source_path is None:
        return None
    sp = Path(source_path)
    candidate_gt_dirs = [
        sp,
        sp / "images" / "DepthImage",
        sp / "DepthImage",
        sp / "images" / "depth",
        sp / "depth",
    ]
    gt_dir = next((d for d in candidate_gt_dirs if _dir_has_exr(d)), None)
    if gt_dir is not None:
        return gt_dir

    try:
        for sub in sorted([p for p in sp.iterdir() if p.is_dir()]):
            if _dir_has_exr(sub):
                return sub
    except FileNotFoundError:
        return None
    return None

def _candidate_pred_depth_dirs(scene_dir: Path, method_dir: Path):
    return [
        method_dir / "depth_exr" / "test",
        method_dir / "depth_exr",
        method_dir / "renders_depth_exr",
        method_dir / "depth_exr_custom",
        scene_dir / "depth_exr_custom",
        scene_dir / "renders_depth_exr",
        scene_dir / "depth_exr" / "test",
        scene_dir / "depth_exr",
    ]

def _resolve_pred_depth_dir(scene_dir: Path, method_dir: Path, pred_depth_dir: str = None) -> Path:
    if pred_depth_dir is not None:
        p = Path(pred_depth_dir)
        return p if p.is_dir() else None
    return next((d for d in _candidate_pred_depth_dirs(scene_dir, method_dir) if d.is_dir()), None)

def _find_gt_depth_for_name(gt_dir: Path, stem: str) -> Path:
    direct = gt_dir / f"{stem}.exr"
    if direct.exists():
        return direct

    m = re.fullmatch(r"\d+", stem)
    if m:
        idx = int(stem)
        try:
            for gt_name in sorted(os.listdir(gt_dir)):
                if not gt_name.lower().endswith(".exr"):
                    continue
                m2 = re.search(r"(\d+)\.exr$", gt_name)
                if m2 and int(m2.group(1)) == idx:
                    return gt_dir / gt_name
        except FileNotFoundError:
            pass
    return None

def _collect_depth_stems(method_dir: Path, depth_dir: Path):
    renders_dir = method_dir / "renders"
    if renders_dir.is_dir():
        return [Path(f).stem for f in sorted(os.listdir(renders_dir)) if not f.startswith(".")]

    stems = []
    for p in sorted(depth_dir.glob("*.exr")):
        stem = p.stem
        if stem.startswith("depth_"):
            stem = stem[len("depth_"):]
        stems.append(stem)
    return stems

def _mean_depth_metrics(depth_metrics_list):
    keys = sorted({k for d in depth_metrics_list for k in d.keys()})
    mean = {}
    for key in keys:
        values = [d[key] for d in depth_metrics_list if np.isfinite(d.get(key, np.nan))]
        mean[key] = float(np.nanmean(values)) if values else float("nan")
    return mean

def _evaluate_depth_pairs(depth_dir: Path, gt_dir: Path, stems, args):
    depth_metrics_list = []
    per_view = {}
    missing = []
    for stem in stems:
        pred_path = _find_depth_pred_for_name(depth_dir, stem)
        gt_path = _find_gt_depth_for_name(gt_dir, stem)
        if pred_path is None or gt_path is None or not gt_path.exists():
            missing.append(stem)
            continue
        try:
            dm = eval_depth_one(
                str(pred_path), str(gt_path),
                pred_channel=args.pred_channel, gt_channel=args.gt_channel,
                pred_scale=args.pred_scale, gt_scale=args.gt_scale,
                min_depth=args.min_depth, max_depth=args.max_depth,
                ratio_outlier=args.ratio_outlier, range_relax=args.range_relax,
                value_clip_method=args.value_clip_method,
                value_clip_p=args.value_clip_p, value_clip_k=args.value_clip_k,
                align=args.align,
            )
            depth_metrics_list.append(dm)
            per_view[stem] = dm
        except Exception as e:
            print(f"[Depth eval fail] {pred_path}: {e}")

    if missing:
        print(f"[Depth eval warning] skipped {len(missing)} unmatched views: {missing[:8]}")
    if not depth_metrics_list:
        return None, per_view
    return _mean_depth_metrics(depth_metrics_list), per_view

def evaluate_depth_only(args):
    for scene_dir_str in args.model_paths:
        scene_dir = Path(scene_dir_str)
        test_dir = scene_dir / "test"
        if test_dir.is_dir():
            method_dirs = [p for p in sorted(test_dir.iterdir()) if p.is_dir()]
        else:
            method_dirs = [scene_dir]

        gt_dir = _resolve_gt_depth_dir(args.source_path)
        if gt_dir is None:
            raise FileNotFoundError(f"Unable to locate GT depth EXR directory from: {args.source_path}")

        out_full = {}
        out_per_view = {}
        for method_dir in method_dirs:
            method = method_dir.name
            depth_dir = _resolve_pred_depth_dir(scene_dir, method_dir, args.pred_depth_dir)
            if depth_dir is None:
                print(f"[Depth eval warning] no predicted depth EXR directory for {method_dir}")
                continue
            stems = _collect_depth_stems(method_dir, depth_dir)
            mean, per_view = _evaluate_depth_pairs(depth_dir, gt_dir, stems, args)
            if mean is None:
                continue
            out_full[method] = mean
            out_per_view[method] = per_view
            print("Scene:", scene_dir)
            print("Method:", method)
            for key in sorted(mean):
                print(f"  {key}: {mean[key]:.8f}")

        if out_full:
            with open(scene_dir / args.output_name, "w") as fp:
                json.dump(out_full, fp, indent=True)
            with open(scene_dir / args.per_view_output_name, "w") as fp:
                json.dump(out_per_view, fp, indent=True)

def _align_shapes(a: np.ndarray, b: np.ndarray, mode: str = "strict"):
    if a.shape == b.shape:
        return a, b
    if mode == "strict":
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if mode == "crop":
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        def crop(x):
            dh = (x.shape[0] - h) // 2
            dw = (x.shape[1] - w) // 2
            return x[dh:dh+h, dw:dw+w]
        return crop(a), crop(b)
    raise ValueError(f"Unknown align mode {mode}")

def _build_base_valid(pred, gt, min_depth, max_depth, ratio_outlier, range_relax=0.05, eps=1e-8):
    valid = np.isfinite(pred) & np.isfinite(gt)
    max_relaxed = max_depth * (1.0 + max(range_relax, 0.0))
    valid &= (gt > min_depth) & (gt <= max_relaxed)
    if not np.any(valid):
        return valid, np.clip(pred, min_depth, max_depth)
    pred_clip = np.clip(pred, min_depth, max_depth)
    valid &= np.isfinite(pred_clip)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum((pred_clip[valid] + eps) / (gt[valid] + eps), (gt[valid] + eps) / (pred_clip[valid] + eps))
    keep = np.zeros_like(valid, bool)
    keep[valid] = ratio <= ratio_outlier
    return valid & keep, pred_clip

def _robust_value_mask(pred_clip, gt, base_valid, method="none", p=99.9, k=3.5, eps=1e-8):
    if not np.any(base_valid) or method == "none":
        return base_valid
    abs_err = np.abs(pred_clip[base_valid] - gt[base_valid])
    absrel = abs_err / (gt[base_valid] + eps)
    keep = np.ones_like(abs_err, bool)
    if method == "percentile":
        thr_rel = np.percentile(absrel, min(max(p, 0.0), 100.0))
        thr_abs = np.percentile(abs_err, min(max(p, 0.0), 100.0))
        keep &= (absrel <= thr_rel)
        keep &= (abs_err <= thr_abs)
    elif method == "mad":
        med = np.median(abs_err)
        mad = np.median(np.abs(abs_err - med)) + eps
        rz = 0.6745 * (abs_err - med) / mad
        keep &= (np.abs(rz) <= k)
    else:
        raise ValueError(f"Unknown outlier method {method}")
    mask = np.zeros_like(base_valid, bool)
    mask[base_valid] = keep
    return base_valid & mask

def _depth_value_metrics(pred_clip, gt, mask, eps=1e-8):
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    diff = pred_clip[mask] - gt[mask]
    abs_diff = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(abs_diff))
    absrel = float(np.mean(abs_diff / (gt[mask] + eps)))
    return rmse, mae, absrel

def _depth_rate_metrics(pred_clip, gt, base_valid, eps=1e-8):
    H, W = pred_clip.shape
    total = H * W
    if total == 0:
        return np.nan, np.nan, np.nan

    cond1 = np.zeros((H, W), dtype=bool)
    cond2 = np.zeros((H, W), dtype=bool)
    cond3 = np.zeros((H, W), dtype=bool)

    if np.any(base_valid):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.maximum((pred_clip[base_valid] + eps) / (gt[base_valid] + eps), (gt[base_valid] + eps) / (pred_clip[base_valid] + eps))
        cond1[base_valid] = ratio < 1.25
        cond2[base_valid] = ratio < 1.25 ** 2
        cond3[base_valid] = ratio < 1.25 ** 3

    d1 = float(cond1.sum() / float(total))
    d2 = float(cond2.sum() / float(total))
    d3 = float(cond3.sum() / float(total))
    return d1, d2, d3

def eval_depth_one(
    pred_path,
    gt_path,
    pred_channel="Z",
    gt_channel="R",
    pred_scale=1.0,
    gt_scale=1.0,
    min_depth=1e-3,
    max_depth=float("inf"),
    ratio_outlier=10.0,
    range_relax=0.05,
    value_clip_method="none",
    value_clip_p=99.9,
    value_clip_k=3.5,
    align="strict",
) -> dict:
    pred = read_exr_channel(pred_path, pred_channel).astype(np.float32) * float(pred_scale)
    gt = read_exr_channel(gt_path, gt_channel).astype(np.float32) * float(gt_scale)
    pred, gt = _align_shapes(pred, gt, mode=align)
    base_valid, pred_clip = _build_base_valid(pred, gt, min_depth, max_depth, ratio_outlier, range_relax)
    value_mask = _robust_value_mask(pred_clip, gt, base_valid, value_clip_method, value_clip_p, value_clip_k)
    rmse, mae, absrel = _depth_value_metrics(pred_clip, gt, value_mask)
    d1, d2, d3 = _depth_rate_metrics(pred_clip, gt, base_valid)
    return {
        "depth.valid_ratio": float(base_valid.mean()),
        "depth.value_ratio": float(value_mask.mean()) if base_valid.any() else 0.0,
        "depth.rmse": rmse,
        "depth.mae": mae,
        "depth.absrel": absrel,
        "depth.delta<1.25": d1,
        "depth.delta<1.25^2": d2,
        "depth.delta<1.25^3": d3,
    }

def evaluate(model_paths, args):

    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims, psnrs, lpipss = [], [], []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item()
                })
                per_view_dict[scene_dir][method].update({
                    "SSIM": {name: s for s, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: p for p, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: l for l, name in zip(torch.tensor(lpipss).tolist(), image_names)}
                })

                # Depth evaluation (if predicted EXR depth exists and source_path is provided)
                # Auto-detect common directories and filename patterns
                if args.source_path is not None:
                    depth_dir = _resolve_pred_depth_dir(Path(scene_dir), method_dir, args.pred_depth_dir)
                    gt_depth_dir = _resolve_gt_depth_dir(args.source_path)
                    if depth_dir is not None and gt_depth_dir is not None:
                        stems = [os.path.splitext(name)[0] for name in image_names]
                        depth_mean, depth_per_view = _evaluate_depth_pairs(depth_dir, gt_depth_dir, stems, args)
                        if depth_mean:
                            full_dict[scene_dir][method].update(depth_mean)
                            per_view_dict[scene_dir][method]["Depth"] = depth_per_view
                    else:
                        print(f"[Depth eval warning] pred depth dir={depth_dir}, gt depth dir={gt_depth_dir}")

            with open(Path(scene_dir) / "results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(Path(scene_dir) / "per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir, "Reason:", e)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--source_path', '-s', required=False, type=str, default=None, help='GT depth directory or dataset root containing images/DepthImage')
    parser.add_argument('--depth-only', action='store_true', help='Only evaluate predicted EXR depth, skip RGB metrics')
    parser.add_argument('--pred-depth-dir', type=str, default=None, help='Optional explicit predicted EXR depth directory')
    parser.add_argument('--output-name', type=str, default='depth_results.json', help='Depth-only summary JSON filename')
    parser.add_argument('--per-view-output-name', type=str, default='depth_per_view.json', help='Depth-only per-view JSON filename')
    parser.add_argument('--pred-channel', type=str, default='Z')
    parser.add_argument('--gt-channel', type=str, default='R')
    parser.add_argument('--pred-scale', type=float, default=1000.0, help='Scale applied to predicted EXR values')
    parser.add_argument('--gt-scale', type=float, default=1000.0, help='Scale applied to GT EXR values')
    parser.add_argument('--min-depth', type=float, default=1e-3)
    parser.add_argument('--max-depth', type=float, default=float('inf'))
    parser.add_argument('--ratio-outlier', type=float, default=10.0)
    parser.add_argument('--range-relax', type=float, default=0.05)
    parser.add_argument('--value-clip-method', choices=['none', 'percentile', 'mad'], default='none')
    parser.add_argument('--value-clip-p', type=float, default=99.9)
    parser.add_argument('--value-clip-k', type=float, default=3.5)
    parser.add_argument('--align', choices=['strict', 'crop'], default='strict')
    args = parser.parse_args()
    if args.depth_only:
        evaluate_depth_only(args)
    else:
        evaluate(args.model_paths, args)
