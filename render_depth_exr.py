#!/usr/bin/env python3
import os
import argparse
from os import makedirs
from typing import Optional

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import MiniCam


def save_depth_exr(depth_m: np.ndarray, out_path: str, far_m: float) -> None:
    """
    Save depth map to EXR with a single Z channel normalized to [0,1].
    Multiply by far_m to recover meters when reading.
    """
    import OpenEXR
    import Imath

    if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
        depth_m = depth_m[:, :, 0]

    far = float(far_m)

    depth_m = np.nan_to_num(depth_m, nan=far, posinf=far, neginf=0.0)
    depth_norm = np.clip(depth_m, 0.0, far) / far
    depth_norm = depth_norm.astype(np.float32)

    h, w = depth_norm.shape
    header = OpenEXR.Header(w, h)
    header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}

    exr = OpenEXR.OutputFile(out_path, header)
    try:
        exr.writePixels({"Z": depth_norm.tobytes()})
    finally:
        exr.close()


@torch.no_grad()
def render_depth_set(model_path: str,
                     name: str,
                     iteration: int,
                     views,
                     gaussians: GaussianModel,
                     pipeline,
                     background: torch.Tensor,
                     use_trained_exp: bool,
                     unit_scale_to_m: float,
                     far_m: float,
                     out_root: Optional[str] = None,
                     target_h: int = 1024,
                     target_w: int = 2048) -> None:
    out_dir = out_root if out_root is not None else os.path.join(model_path, name, f"ours_{iteration}", "depth_exr")
    makedirs(out_dir, exist_ok=True)

    for view in views:
        # Wrap view into a MiniCam with forced resolution (no internal downsampling)
        cam = MiniCam(
            width=int(target_w),
            height=int(target_h),
            fovy=view.FoVy,
            fovx=view.FoVx,
            znear=view.znear,
            zfar=view.zfar,
            world_view_transform=view.world_view_transform,
            full_proj_transform=view.full_proj_transform,
        )
        # Preserve image name for exposure lookup compatibility
        setattr(cam, "image_name", getattr(view, "image_name", "view"))

        pkg = render(cam, gaussians, pipeline, background, use_trained_exp=use_trained_exp, separate_sh=False)

        # Rasterizer returns inverse-depth proxy in pkg["depth"]. Convert to linear depth.
        depth_inv = pkg["depth"].detach()  # [H,W]
        depth_lin = 1.0 / (depth_inv + 1e-5)

        depth_m = (depth_lin * float(unit_scale_to_m)).clamp(min=0.0)
        depth_np = depth_m.squeeze().cpu().numpy().astype(np.float32)

        stem, _ = os.path.splitext(view.image_name)
        out_path = os.path.join(out_dir, f"depth_{stem}.exr")
        save_depth_exr(depth_np, out_path, far_m=far_m)


def main() -> None:
    # 这里描述改一下：默认同时渲染 train + test
    parser = argparse.ArgumentParser(
        description="Render 3D-UIR depth to EXR (Z channel normalized). Default: render TRAIN and TEST."
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Checkpoint iteration to load; -1 = latest")

    # ======= 关键修改：默认不跳过 train/test，跑脚本时会同时渲染两套 =======
    parser.add_argument(
        "--skip-train",
        action="store_true",
        default=False,           # 原来是 True
        help="Skip rendering train views (default: render train views)"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        default=False,           # 显式设为 False，默认不跳过 test
        help="Skip rendering test views (default: render test views)"
    )
    # 以前的 --render-train 现在已经不需要了，如要保留也可以，但这里简单直接删掉
    # ===============================================================

    parser.add_argument("--unit-scale-to-m", type=float, default=1.0, help="Scene unit to meters scale (meters per scene unit)")
    parser.add_argument("--far-m", type=float, default=1000.0, help="Far depth in meters used for EXR normalization")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory root; defaults to model_path/<set>/ours_<it>/depth_exr")
    parser.add_argument("--height", type=int, default=1024, help="Output height in pixels (default 1024)")
    parser.add_argument("--width", type=int, default=2048, help="Output width in pixels (default 2048)")
    args = get_combined_args(parser)

    print("Rendering depth EXR for", args.model_path)

    safe_state(False)

    with torch.no_grad():
        gaussians = GaussianModel(model.extract(args).sh_degree)
        scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)

        bg_color = [1, 1, 1] if model.extract(args).white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        outdir = getattr(args, "outdir", None)

        # 如果用户没有传 --skip-train，就会执行这里
        if not args.skip_train:
            render_depth_set(
                model.extract(args).model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline.extract(args),
                background,
                model.extract(args).train_test_exp,
                unit_scale_to_m=args.unit_scale_to_m,
                far_m=args.far_m,
                out_root=outdir,
                target_h=args.height,
                target_w=args.width,
            )

        # 如果用户没有传 --skip-test，就会执行这里
        if not args.skip_test:
            render_depth_set(
                model.extract(args).model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline.extract(args),
                background,
                model.extract(args).train_test_exp,
                unit_scale_to_m=args.unit_scale_to_m,
                far_m=args.far_m,
                out_root=outdir,
                target_h=args.height,
                target_w=args.width,
            )


if __name__ == "__main__":
    main()
