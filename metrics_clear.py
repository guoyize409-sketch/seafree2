#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估“去除水体”后的 RGB 图像质量指标（PSNR / SSIM / LPIPS）

使用方式示例：
  python metrics_clear.py \
    --pred-dir /home/gyz/3D-UIR/output/9ee1baa1-f/test/ours_30000/clear \
    --gt-dir /mnt/d/3009/home/ycy/oceanEnvironmentPark/with/air \
    --pred-glob \"*.png\" \
    --align resize \
    --output eval_rgb_clear_results.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from PIL import Image

# 可选：lpips（pip install lpips）
_lpips_model = None
try:
    import lpips

    _lpips_model = lpips.LPIPS(net="vgg")
    if torch.cuda.is_available():
        _lpips_model = _lpips_model.cuda()
    _lpips_model.eval()
    print("[INFO] 已加载 LPIPS 模型 (VGG)")
except ImportError:
    print("[WARN] 未安装 lpips，LPIPS 指标将不可用（将返回 NaN），请运行: pip install lpips")
    _lpips_model = None
except Exception as e:
    print(f"[WARN] LPIPS 初始化失败: {e}")
    _lpips_model = None

# 优先使用项目内实现；否则降级为简化实现
try:
    from utils.loss_utils import ssim  # type: ignore
    from utils.image_utils import psnr  # type: ignore
except Exception:
    def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
        """简化的 SSIM 实现（与 metrics3.py 中降级实现一致）"""
        from torch.nn.functional import conv2d
        import torch.nn as nn

        def gaussian(window_size: int, sigma: float) -> torch.Tensor:
            gauss = torch.Tensor(
                [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
            )
            return gauss / gauss.sum()

        def create_window(window_size: int, channel: int) -> torch.Tensor:
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window

        channel = img1.size(1)
        window = create_window(window_size, channel).to(img1.device)

        mu1 = conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """简化的 PSNR 实现（与 metrics3.py 中降级实现一致），输入应为 [0,1]"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _align_tensors_center_crop(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将两个 4D 张量（N,C,H,W）中心裁剪到相同的最小 H,W。
    """
    assert a.dim() == 4 and b.dim() == 4, "输入必须为 4D 张量 (N,C,H,W)"
    _, _, ha, wa = a.shape
    _, _, hb, wb = b.shape
    h = min(ha, hb)
    w = min(wa, wb)
    if ha != h or wa != w:
        a = tf.center_crop(a, [h, w])
    if hb != h or wb != w:
        b = tf.center_crop(b, [h, w])
    return a, b


def _resize_tensor(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    以双线性插值将 4D 张量 (N,C,H,W) 调整为给定 (H,W)。
    """
    assert x.dim() == 4, "输入必须为 4D 张量 (N,C,H,W)"
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def eval_rgb(pred_path: str, gt_path: str, align: str = "strict") -> Dict[str, float]:
    """
    评估 RGB 图像：PSNR / SSIM / LPIPS
    - 输入图像将被转换为张量，范围 [0,1]
    - 当 align='crop' 时，若尺寸不同，使用中心裁剪到相同大小；
    - 当 align='resize' 时，若尺寸不同，将预测图像缩放到 GT 尺寸；
    - align='strict' 则要求尺寸一致
    """
    pred_img = Image.open(pred_path)
    gt_img = Image.open(gt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_tensor = tf.to_tensor(pred_img).unsqueeze(0)[:, :3, :, :].to(device)
    gt_tensor = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].to(device)

    if pred_tensor.shape[-2:] != gt_tensor.shape[-2:]:
        if align == "crop":
            pred_tensor, gt_tensor = _align_tensors_center_crop(pred_tensor, gt_tensor)
        elif align == "resize":
            target_h, target_w = gt_tensor.shape[-2], gt_tensor.shape[-1]
            pred_tensor = _resize_tensor(pred_tensor, (target_h, target_w))
        else:
            raise ValueError(f"图像尺寸不一致且未允许对齐: {pred_tensor.shape[-2:]} vs {gt_tensor.shape[-2:]}")

    results = {}
    results["psnr"] = float(psnr(pred_tensor, gt_tensor))
    results["ssim"] = float(ssim(pred_tensor, gt_tensor))

    if _lpips_model is not None:
        with torch.no_grad():
            pred_lpips = pred_tensor * 2.0 - 1.0
            gt_lpips = gt_tensor * 2.0 - 1.0
            lpips_value = _lpips_model(pred_lpips, gt_lpips)
            results["lpips"] = float(lpips_value.item())
    else:
        results["lpips"] = float("nan")

    return results


def find_matching_gt_file(pred_file: Path, gt_dir: Path, gt_index: Optional[int] = None) -> Optional[Path]:
    """
    根据预测文件名在真值目录中查找最合适的匹配文件。
    逻辑参考 metrics3.py 中的实现，包含多种命名兼容与 images/ColorImage 的降级路径。
    """
    pred_stem = pred_file.stem
    pred_suffix = pred_file.suffix

    # 如果 gt_dir 是文件，直接返回
    if gt_dir.is_file():
        return gt_dir

    # 若传入的目录不存在，尝试 images/ColorImage 降级
    if not gt_dir.exists():
        parent = gt_dir.parent
        if parent.exists():
            color_image_dir = parent / "images" / "ColorImage"
            if color_image_dir.exists():
                gt_dir = color_image_dir

    if not gt_dir.exists():
        return None

    # 若提供 gt_index，先尝试按索引取文件
    if gt_index is not None:
        candidates = [
            gt_dir / f"{gt_index:04d}.png",
            gt_dir / f"{gt_index:04d}.jpg",
            gt_dir / f"{gt_index}.png",
            gt_dir / f"{gt_index}.jpg",
            gt_dir / f"image_{gt_index:04d}.png",
            gt_dir / f"frame_{gt_index:04d}.png",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        all_files = sorted([p for p in gt_dir.glob("*") if p.is_file()])
        if 0 <= gt_index < len(all_files):
            return all_files[gt_index]

    # 直接同名匹配
    candidates = [
        gt_dir / f"{pred_stem}{pred_suffix}",
        gt_dir / f"{pred_stem}.png",
        gt_dir / f"{pred_stem}.jpg",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    # 若预测文件名包含数字，基于最后一个数字匹配。
    # Seafree 扁平输出形如 eval_intrinsic_color_render_0014.png。
    matches = re.findall(r"(\d+)", pred_stem)
    match = matches[-1] if matches else None
    if match:
        idx = match
        candidates = [
            gt_dir / f"{idx}.png",
            gt_dir / f"{idx}.jpg",
            gt_dir / f"image_{idx}.png",
            gt_dir / f"frame_{idx}.png",
            gt_dir / f"{int(idx):04d}.png",
            gt_dir / f"{int(idx):05d}.png",
        ]
        for cand in candidates:
            if cand.exists():
                return cand

    # 兜底：按排序索引匹配
    all_files = sorted([p for p in gt_dir.glob("*") if p.is_file()])
    if match:
        idx_int = int(match)
        if 0 <= idx_int < len(all_files):
            return all_files[idx_int]
    if len(all_files) == 1:
        return all_files[0]

    return None


def list_pred_images(pred_dir: Path, pred_glob: str) -> List[Path]:
    """
    列出预测目录下的图像文件（按名字排序）
    默认只评估常见图像格式；可通过 pred_glob 指定更精确的模式（如 \"*.png\"）。
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    seafree_intrinsic_exact = re.compile(
        r"^eval_intrinsic_color_render_\d+\.(png|jpg|jpeg|bmp|tif|tiff)$",
        re.IGNORECASE,
    )
    files = []
    for p in sorted(pred_dir.glob(pred_glob)):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        # SeaFree 同目录下可能同时存在 eval_intrinsic_color_render_raw_*.png。
        # auto 模式只评估最终可视化输出，避免 raw 图被重复计入指标。
        if pred_glob == "eval_intrinsic_color_render_*.png" and not seafree_intrinsic_exact.match(p.name):
            continue
        files.append(p)
    return files


def list_gt_images(gt_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not gt_dir.exists() or not gt_dir.is_dir():
        return []
    return sorted([p for p in gt_dir.glob("*") if p.is_file() and p.suffix.lower() in exts])


def resolve_pred_input(pred_input: Path, pred_glob: str) -> Tuple[Path, str]:
    """
    Resolve normal prediction directories and Seafree eval_renders flat prefixes.

    Supported examples:
      - /path/to/clear
      - /path/to/eval_renders with --pred-glob auto
      - /path/to/eval_renders/eval_intrinsic_color_render
    """
    if pred_input.exists():
        if pred_input.is_file():
            return pred_input.parent, pred_input.name
        if pred_glob != "auto":
            return pred_input, pred_glob
        seafree_glob = "eval_intrinsic_color_render_*.png"
        if list_pred_images(pred_input, seafree_glob):
            return pred_input, seafree_glob
        return pred_input, "*.png"

    parent = pred_input.parent
    prefix = pred_input.name
    if parent.exists() and prefix:
        prefix_glob = f"{prefix}_*.png"
        if list_pred_images(parent, prefix_glob):
            return parent, prefix_glob
        loose_glob = f"{prefix}*.png"
        if list_pred_images(parent, loose_glob):
            return parent, loose_glob
    return pred_input, pred_glob


def default_output_path(pred_dir: Path, pred_glob: str) -> Path:
    if pred_glob.startswith("eval_intrinsic_color_render"):
        return pred_dir / "clear_results.json"
    return pred_dir / "eval_rgb_clear_results.json"


def infer_gt_index_stride(pred_files: List[Path], gt_dir: Path, pred_glob: str, stride_arg: str) -> Optional[int]:
    if stride_arg not in ("auto", "", None):
        if stride_arg.lower() in ("none", "name"):
            return None
        stride = int(stride_arg)
        return stride if stride > 0 else None

    if not pred_glob.startswith("eval_intrinsic_color_render"):
        return None

    gt_files = list_gt_images(gt_dir)
    if len(pred_files) == 0 or len(gt_files) <= len(pred_files):
        return None
    if len(gt_files) % len(pred_files) != 0:
        return None
    stride = len(gt_files) // len(pred_files)
    return stride if stride > 1 else None


def compute_mean_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    计算多张图的均值指标（忽略 NaN / 非有限值）
    """
    if not all_metrics:
        return {}
    keys = set()
    for m in all_metrics:
        keys.update(k for k, v in m.items() if isinstance(v, (int, float)) and np.isfinite(v))
    mean_result: Dict[str, float] = {}
    for k in keys:
        vals = []
        for m in all_metrics:
            if k in m and isinstance(m[k], (int, float)) and np.isfinite(m[k]):
                vals.append(float(m[k]))
        mean_result[k] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    return mean_result


def main() -> int:
    parser = argparse.ArgumentParser(description="评估去除水体的 RGB 图像（PSNR/SSIM/LPIPS）")
    parser.add_argument("--pred-dir", type=str, required=True, help="预测图像目录；也可传 Seafree eval_renders/eval_intrinsic_color_render 前缀")
    parser.add_argument("--gt-dir", type=str, required=True, help="真实/空气图像目录（或该目录下的 images/ColorImage）")
    parser.add_argument("--pred-glob", type=str, default="auto", help="预测文件匹配模式；auto 会优先匹配 Seafree 的 eval_intrinsic_color_render_数字.png，并排除 raw 输出")
    parser.add_argument("--gt-index-stride", type=str, default="auto",
                        help="GT 索引步长；auto 对 Seafree eval_intrinsic_color_render_*.png 自动使用 GT数量/预测数量，例如 120/15=8；none 表示仅按文件名匹配")
    parser.add_argument("--align", type=str, default="strict", choices=["strict", "crop", "resize"],
                        help="尺寸不一致时的对齐策略：strict=直接报错；crop=中心裁剪；resize=将预测缩放到GT尺寸")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件路径（默认写入预测目录）")
    args = parser.parse_args()

    pred_input = Path(args.pred_dir).resolve()
    pred_dir, pred_glob = resolve_pred_input(pred_input, args.pred_glob)
    gt_dir = Path(args.gt_dir).resolve()

    if not pred_dir.exists():
        print(f"[ERROR] 预测目录/前缀不存在: {pred_input}")
        return 1
    if not gt_dir.exists():
        # 允许稍后在 find_matching_gt_file 内部尝试 images/ColorImage 降级
        print(f"[WARN] 真值目录不存在（将尝试降级路径）: {gt_dir}")

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = default_output_path(pred_dir, pred_glob)

    pred_files = list_pred_images(pred_dir, pred_glob)
    if len(pred_files) == 0:
        print(f"[ERROR] 在 {pred_dir} 下未找到任何匹配 {pred_glob} 的图像文件")
        return 1
    gt_index_stride = infer_gt_index_stride(pred_files, gt_dir, pred_glob, args.gt_index_stride)

    print(f"[INFO] 预测目录: {pred_dir}")
    print(f"[INFO] 预测匹配: {pred_glob}")
    print(f"[INFO] 真值目录: {gt_dir}")
    print(f"[INFO] 待评估图像数: {len(pred_files)}")
    if gt_index_stride is not None:
        print(f"[INFO] GT 索引步长: {gt_index_stride}")

    per_image_results: Dict[str, Dict[str, float]] = {}
    all_metrics: List[Dict[str, float]] = []

    for i, pred_file in enumerate(pred_files):
        print("\n" + "=" * 80)
        print(f"评估第 {i + 1}/{len(pred_files)} 张: {pred_file.name}")
        gt_index = i * gt_index_stride if gt_index_stride is not None else None
        gt_file = find_matching_gt_file(pred_file, gt_dir, gt_index=gt_index)
        if gt_file is None:
            print(f"[WARN] 未能在 {gt_dir}（及其可能的 images/ColorImage）为 {pred_file.name} 匹配到 GT 图像")
            per_image_results[pred_file.name] = {"error": "未找到匹配的真值图像"}
            continue

        try:
            metrics = eval_rgb(str(pred_file), str(gt_file), align=args.align)
            metrics["gt_file"] = str(gt_file)
            per_image_results[pred_file.name] = metrics
            all_metrics.append(metrics)
            print(f"PSNR: {metrics['psnr']:.4f}, SSIM: {metrics['ssim']:.4f}, LPIPS: {metrics['lpips']:.4f}")
        except Exception as e:
            per_image_results[pred_file.name] = {"error": str(e)}
            print(f"[ERROR] 评估失败: {e}")

    mean_metrics = compute_mean_metrics(all_metrics)

    results = {
        "count": len(pred_files),
        "matched_count": len(all_metrics),
        "pred_dir": str(pred_dir),
        "pred_glob": pred_glob,
        "gt_dir": str(gt_dir),
        "gt_index_stride": gt_index_stride,
        "per_image": per_image_results,
        "mean": mean_metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    if mean_metrics:
        print(f"总体均值 - PSNR: {mean_metrics.get('psnr', float('nan')):.4f}, "
              f"SSIM: {mean_metrics.get('ssim', float('nan')):.4f}, "
              f"LPIPS: {mean_metrics.get('lpips', float('nan')):.4f}")
    print(f"评估完成！结果已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
