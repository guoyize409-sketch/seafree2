#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
将 Z:\AAAAA_UE 下的 EXR 深度图批量转换为 PNG，并把输出写入一个你指定的输出根目录，
保持与输入根目录完全一致的相对路径结构。兼容 SeaFree-GS 的 depth_exr/test 输出。
输出路径的最后一级目录名强制为 DepthImage。

默认参数对齐单文件可视化脚本：
--map equalize  --invert 1  --gamma 0.9  --low_q 1  --high_q 99  --colormap JET

具备断点续转能力：
- 默认跳过已存在且不比源 EXR 旧的 PNG
- 如需重写全部，使用 --overwrite

用法示例：
python exr_depth_mirror_export.py "Z:\AAAAA_UE" "D:\UE_depth_png"
python exr_depth_mirror_export.py "Z:\AAAAA_UE" "D:\UE_depth_png" --dry-run
python exr_depth_mirror_export.py "Z:\AAAAA_UE" "D:\UE_depth_png" --overwrite
"""

import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import os
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# 读取 EXR 依赖
try:
    import OpenEXR, Imath
except Exception:
    OpenEXR = None
    Imath = None

# 写 PNG 依赖
_USE_IMAGEIO = True
try:
    import imageio.v3 as iio
except Exception:
    _USE_IMAGEIO = False
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("需要 imageio 或 pillow 以写 PNG")

# OpenCV 用于着色与 CLAHE
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    raise RuntimeError("需要 OpenCV，pip install opencv-python")

# ---------- EXR 读取 ----------
def _pick_depth_channel(header, preferred="R") -> str:
    chans = list(header.get("channels", {}).keys())
    lower = {c.lower(): c for c in chans}
    for cand in [preferred, "Z", "Depth", "G", "B"]:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if not chans:
        raise RuntimeError("EXR 中不含任何通道")
    return chans[0]

def read_exr_depth(path: Path, channel_hint: str = "R") -> np.ndarray:
    if OpenEXR is None:
        if not _HAS_CV2:
            raise RuntimeError("需要 OpenEXR/Imath 或 OpenCV EXR 支持来读取 EXR")
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"OpenCV 读取 EXR 失败: {path}")
        if arr.ndim == 3:
            requested = channel_hint.lower()
            channel_map = {"b": 0, "g": 1, "r": 2, "z": 0, "depth": 0}
            arr = arr[..., channel_map.get(requested, 0)]
        return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    f = OpenEXR.InputFile(str(path))
    header = f.header()
    dw = header["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    ch = _pick_depth_channel(header, preferred=channel_hint)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    arr = np.frombuffer(f.channel(ch, FLOAT), dtype=np.float32).reshape(H, W)
    f.close()
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

# ---------- 工具 ----------
def percentile_clip(x, low_q, high_q):
    lo = np.percentile(x, low_q)
    hi = np.percentile(x, high_q)
    if hi <= lo:
        hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return y, lo, hi

def global_equalize_01(x01):
    flat = x01.ravel()
    order = np.argsort(flat)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, endpoint=True)
    return ranks.reshape(x01.shape)

# ---------- 映射与着色 ----------
def depth_to_color(
    depth_norm: np.ndarray,
    near_m: float = 0.0,
    far_m: float = 1000.0,
    low_q: float = 1.0,
    high_q: float = 99.0,
    map_mode: str = "equalize",   # 'equalize' | 'sigmoid' | 'linear'
    gain: float = 10.0,
    gamma: float = 0.9,
    invert_flag: int = 1,         # 1 表示近红远蓝，0 表示不反转
    local_eq: int = 0,
    colormap: str = "JET",
    keep_linear: bool = False
):
    if not _HAS_CV2:
        raise RuntimeError("需要 opencv-python 以进行着色和可选 CLAHE")

    invert = bool(invert_flag)
    local_eq_bool = bool(local_eq)

    valid = np.isfinite(depth_norm) & (depth_norm > 0)
    depth_m = np.zeros_like(depth_norm, np.float32)
    depth_m[valid] = np.clip(depth_norm[valid] * 1000.0, near_m, far_m)

    depth01 = np.zeros_like(depth_m, np.float32)
    if np.any(valid):
        d = depth_m[valid] / max(far_m, 1e-6)
        d_clip, lo, hi = percentile_clip(d, low_q, high_q)

        if map_mode == "equalize":
            y = global_equalize_01(d_clip)
        elif map_mode == "sigmoid":
            c = np.median(d_clip)
            z = (d_clip - c) / max(hi - lo, 1e-6)
            y = 1.0 / (1.0 + np.exp(-gain * z))
            y = (y - y.min()) / (y.max() - y.min() + 1e-12)
        else:
            y = d_clip

        if local_eq_bool:
            u8 = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            u8 = clahe.apply(u8)
            y = u8.astype(np.float32) / 255.0

        y = np.clip(y, 0.0, 1.0)
        if not keep_linear:
            y = y ** max(gamma, 1e-6)
        if invert:
            y = 1.0 - y

        depth01[valid] = y

    u8 = (depth01 * 255.0 + 0.5).astype(np.uint8)
    cmap = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS
    }.get(colormap.upper(), cv2.COLORMAP_JET)
    color_bgr = cv2.applyColorMap(u8, cmap)
    if np.any(~valid):
        color_bgr[~valid] = (0, 0, 0)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return color_rgb

# ---------- 写 PNG ----------
def write_png(path: Path, arr_u8_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _USE_IMAGEIO:
        iio.imwrite(str(path), arr_u8_rgb)
    else:
        Image.fromarray(arr_u8_rgb, mode="RGB").save(str(path))

# ---------- 断点续转判定 ----------
def png_is_fresh(png_path: Path, exr_path: Path) -> bool:
    """若 PNG 存在、大小大于 0 且 mtime 不早于 EXR，则视为已转好。"""
    try:
        if not png_path.exists():
            return False
        if png_path.stat().st_size <= 0:
            return False
        return png_path.stat().st_mtime >= exr_path.stat().st_mtime
    except Exception:
        return False

# ---------- 单个文件转换 ----------
def convert_one(
    exr_path: Path,
    png_path: Path,
    channel: str = "R",
    near_m: float = 0.0,
    far_m: float = 1000.0,
    low_q: float = 1.0,
    high_q: float = 99.0,
    map_mode: str = "equalize",
    gain: float = 10.0,
    gamma: float = 0.9,
    invert_flag: int = 1,
    local_eq: int = 0,
    colormap: str = "JET",
    keep_linear: bool = False
):
    depth_norm = read_exr_depth(exr_path, channel_hint=channel)
    color_rgb = depth_to_color(
        depth_norm,
        near_m=near_m, far_m=far_m,
        low_q=low_q, high_q=high_q,
        map_mode=map_mode, gain=gain, gamma=gamma,
        invert_flag=invert_flag, local_eq=local_eq,
        colormap=colormap, keep_linear=keep_linear
    )
    write_png(png_path, color_rgb)

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser(description="镜像导出 EXR 深度到 PNG，保持目录结构一致，叶子目录名为 DepthImage")
    ap.add_argument("input_root", type=str, help="输入根目录，例如 Z:\\AAAAA_UE")
    ap.add_argument("output_root", type=str, help="输出根目录，将镜像生成 PNG")
    # 深度映射参数，默认与单文件脚本对齐
    ap.add_argument("--channel", default="R", choices=["R","G","B","Z"])
    ap.add_argument("--near_m", type=float, default=0.0)
    ap.add_argument("--far_m", type=float, default=1000.0)
    ap.add_argument("--low_q", type=float, default=1.0)
    ap.add_argument("--high_q", type=float, default=99.0)
    ap.add_argument("--map", choices=["equalize","sigmoid","linear"], default="equalize")
    ap.add_argument("--gain", type=float, default=10.0)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--invert", type=int, default=1, help="近红远蓝开关，1 开启，0 关闭")
    ap.add_argument("--local_eq", type=int, default=0, help="在深度域做 CLAHE 局部均衡，1 开启")
    ap.add_argument("--colormap", default="JET")
    ap.add_argument("--keep-linear", action="store_true", help="保持线性，不做 γ 校正")
    # 断点续转与覆盖策略
    ap.add_argument("--overwrite", action="store_true", help="强制重写已存在 PNG，不做断点续转")
    ap.add_argument("--skip-test-dir", action="store_true", help="跳过输入根目录下名为 test 的一级子目录（保留旧数据集行为）")
    ap.add_argument("--dry-run", action="store_true", help="仅打印计划，不实际写文件")
    args = ap.parse_args()

    in_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    if not in_root.exists():
        raise FileNotFoundError(f"输入根目录不存在: {in_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = [p for p in in_root.iterdir() if p.is_dir()]
    # 若输入根目录下没有子目录，或 EXR 直接放在根下，则也处理根目录
    if not scenes:
        scenes = [in_root]
    total, done, skipped = 0, 0, 0

    for scene in sorted(scenes):
        if args.skip_test_dir and scene != in_root and scene.name.lower() == "test":
            print(f"[跳过] {scene}")
            continue

        # 收集 EXR：优先匹配默认路径，其次回退为全局匹配，覆盖“根目录直接放 EXR”的情况
        exr_candidates = []
        for pat in ("images/DepthImage/*.exr", "**/*.exr"):
            exr_candidates.extend(scene.rglob(pat))
        # 去重并保持顺序
        seen = set()
        exr_list = []
        for p in exr_candidates:
            key = p
            try:
                key = p.resolve()
            except Exception:
                pass
            if key not in seen:
                seen.add(key)
                exr_list.append(p)
        if not exr_list:
            continue

        print(f"[发现] {scene.name}  EXR 数量 {len(exr_list)}")
        total += len(exr_list)

        for exr_path in exr_list:
            rel = exr_path.relative_to(in_root)
            parts = list(rel.parts)
            if len(parts) >= 2:
                parts[-2] = "DepthImage"
            rel_depth = Path(*parts)
            png_path = (out_root / rel_depth).with_suffix(".png")

            # 断点续转判定
            if not args.overwrite and png_is_fresh(png_path, exr_path):
                if args.dry_run:
                    print(f"  [已存在] 跳过  {exr_path}  ->  {png_path}")
                else:
                    print(f"  [已存在] 跳过  {png_path}")
                skipped += 1
                continue

            if args.dry_run:
                print(f"  {exr_path}  ->  {png_path}")
                continue

            try:
                convert_one(
                    exr_path,
                    png_path,
                    channel=args.channel,
                    near_m=args.near_m, far_m=args.far_m,
                    low_q=args.low_q, high_q=args.high_q,
                    map_mode=args.map, gain=args.gain, gamma=args.gamma,
                    invert_flag=args.invert, local_eq=args.local_eq,
                    colormap=args.colormap, keep_linear=args.keep_linear
                )
                done += 1
            except Exception as e:
                print(f"[失败] {exr_path}  原因: {e}")

    if args.dry_run:
        print(f"\n计划处理 {total} 个 EXR 文件，其中将跳过 {skipped} 个已完成，预计转换 {total - skipped} 个")
    else:
        print(f"\n完成转换。共找到 {total} 个 EXR，跳过 {skipped} 个已完成，成功输出 {done} 个 PNG 到 {out_root}")

if __name__ == "__main__":
    main()
