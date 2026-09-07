#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SeaFree/Nerfstudio raw depth to normalized EXR."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to eval_renders, scene output root, metrics_test.json, or config.yml.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "train+test"],
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root. Default: <eval_renders>/depth_exr when input is eval_renders, otherwise <scene>/depth_exr.",
    )
    parser.add_argument("--near", type=float, default=1e-8, help="Minimum positive depth kept in EXR.")
    parser.add_argument("--far-m", type=float, default=1000.0, help="Depth normalization range in meters.")
    parser.add_argument(
        "--output-mode",
        type=str,
        default="normalized",
        choices=["normalized", "meters"],
        help="Write depth normalized by far-m or raw meters.",
    )
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing EXR files.")
    return parser.parse_args()


def infer_project_root(config_path: Path) -> Optional[Path]:
    for parent in config_path.parents:
        if parent.name.startswith("outputs"):
            return parent.parent
    return None


def prefer_config_project_code(config_path: Path) -> None:
    project_root = infer_project_root(config_path)
    if project_root is None:
        return
    for path in [project_root, project_root / "third_party" / "gsplat"]:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def _config_from_metrics(metrics_path: Path) -> Path:
    data = json.loads(metrics_path.read_text())
    checkpoint = Path(data["checkpoint"]).resolve()
    config_path = checkpoint.parent.parent / "config.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Unable to locate config.yml from checkpoint: {checkpoint}")
    return config_path


def _find_latest_config(scene_root: Path) -> Optional[Path]:
    configs = sorted(scene_root.glob("seafree-gs/*/config.yml"))
    if configs:
        return configs[-1]
    return None


def resolve_config_path(input_path: Path) -> Path:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.name == "config.yml":
            return input_path
        if input_path.name == "metrics_test.json":
            return _config_from_metrics(input_path)
        raise ValueError(f"Unsupported file input: {input_path}")

    candidates = [
        input_path / "metrics_test.json",
        input_path / "config.yml",
    ]
    if input_path.name == "eval_renders":
        candidates.insert(0, input_path.parent / "metrics_test.json")
        latest = _find_latest_config(input_path.parent)
    else:
        latest = _find_latest_config(input_path)
    for candidate in candidates:
        if candidate.is_file():
            return resolve_config_path(candidate)
    if latest is not None:
        return latest.resolve()
    raise FileNotFoundError(f"Unable to resolve config.yml from: {input_path}")


def infer_scene_root(input_path: Path, config_path: Path) -> Path:
    input_path = input_path.resolve()
    if input_path.is_dir() and input_path.name == "eval_renders":
        return input_path.parent
    if input_path.is_dir() and (input_path / "eval_renders").is_dir():
        return input_path
    if input_path.is_file() and input_path.name == "metrics_test.json":
        return input_path.parent

    for parent in config_path.parents:
        if parent.name.startswith("outputs"):
            try:
                rel = config_path.relative_to(parent)
                if len(rel.parts) >= 2:
                    return parent / rel.parts[0] / rel.parts[1]
            except ValueError:
                pass
    raise ValueError(f"Unable to infer scene root from: {config_path}")


def default_output_root(input_path: Path, scene_root: Path, output_root: Optional[str]) -> Path:
    if output_root is not None:
        return Path(output_root).resolve()
    input_path = input_path.resolve()
    if input_path.is_dir() and input_path.name == "eval_renders":
        return (input_path / "depth_exr").resolve()
    return (scene_root / "depth_exr").resolve()


def save_exr(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = np.asarray(depth, dtype=np.float32)
    ok = cv2.imwrite(str(path), depth)
    if not ok:
        raise RuntimeError(f"Failed to write EXR: {path}")


def sanitize_depth(depth: torch.Tensor, near: float) -> np.ndarray:
    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    depth_np[depth_np < float(near)] = 0.0
    return depth_np


def format_depth_for_output(depth_np: np.ndarray, output_mode: str, far_m: float) -> np.ndarray:
    if output_mode == "meters":
        return depth_np
    far_m = max(float(far_m), 1e-8)
    return np.clip(depth_np, 0.0, far_m) / far_m


def get_loaded_eval_datamanager(pipeline):
    datamanager = getattr(pipeline, "datamanager", None)
    if datamanager is None:
        return None
    dataset = getattr(datamanager, "eval_dataset", None)
    dataloader = getattr(datamanager, "fixed_indices_eval_dataloader", None)
    if dataset is None or dataloader is None:
        return None
    dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
    if dataparser_outputs is None:
        dataparser_outputs = getattr(datamanager, "eval_dataparser_outputs", None)
    if dataparser_outputs is None:
        dataparser = getattr(datamanager, "dataparser", None)
        test_split = getattr(datamanager, "test_split", "test")
        if dataparser is not None:
            dataparser_outputs = dataparser.get_dataparser_outputs(split=test_split)
    if dataparser_outputs is None:
        return None
    return dataset, dataloader, dataparser_outputs


def build_datamanager(config, pipeline_device: torch.device, split: str):
    from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
    from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
    from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
    from nerfstudio.scripts.render import _disable_datamanager_setup

    data_manager_config = config.pipeline.datamanager
    assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))

    with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
        datamanager = data_manager_config.setup(test_mode=split, device=pipeline_device)

    if split == "train":
        dataset = datamanager.train_dataset
        dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
    else:
        dataset = datamanager.eval_dataset
        dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
        if dataparser_outputs is None:
            dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)

    dataloader = FixedIndicesEvalDataloader(
        input_dataset=dataset,
        device=datamanager.device,
        num_workers=datamanager.world_size * 4,
    )
    return dataset, dataloader, dataparser_outputs


def render_split(
    pipeline,
    config,
    split: str,
    out_dir: Path,
    near: float,
    far_m: float,
    output_mode: str,
    overwrite: bool,
) -> int:
    if split == "test":
        loaded = get_loaded_eval_datamanager(pipeline)
        if loaded is not None:
            _dataset, dataloader, dataparser_outputs = loaded
        else:
            _dataset, dataloader, dataparser_outputs = build_datamanager(config, pipeline.device, split)
    else:
        _dataset, dataloader, dataparser_outputs = build_datamanager(config, pipeline.device, split)

    image_filenames = [Path(os.path.normpath(str(p))) for p in dataparser_outputs.image_filenames]
    images_root = Path(os.path.commonpath([str(p) for p in image_filenames]))
    dataparser_scale = float(dataparser_outputs.dataparser_scale)

    written = 0
    with torch.no_grad():
        for camera_idx, (camera, _batch) in enumerate(dataloader):
            image_path = image_filenames[camera_idx]
            try:
                image_name = image_path.relative_to(images_root)
            except ValueError:
                image_name = Path(image_path.name)
            out_path = (out_dir / image_name).with_suffix(".exr")
            if out_path.exists() and not overwrite:
                continue

            outputs = pipeline.model.get_outputs_for_camera(camera)
            if "depth" not in outputs:
                raise KeyError("Model output does not contain 'depth'.")

            depth = outputs["depth"] / dataparser_scale
            depth_np = sanitize_depth(depth, near=near)
            depth_np = format_depth_for_output(depth_np, output_mode=output_mode, far_m=far_m)
            save_exr(out_path, depth_np)
            written += 1

    return written


def configure_eval_dataset(config):
    from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
    from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig

    data_manager_config = config.pipeline.datamanager
    assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))
    data_manager_config.eval_num_images_to_sample_from = -1
    data_manager_config.eval_num_times_to_repeat_images = -1
    if isinstance(data_manager_config, VanillaDataManagerConfig):
        data_manager_config.train_num_images_to_sample_from = -1
        data_manager_config.train_num_times_to_repeat_images = -1
    return config


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    config_path = resolve_config_path(input_path)
    prefer_config_project_code(config_path)

    from nerfstudio.utils.eval_utils import eval_setup

    scene_root = infer_scene_root(input_path, config_path)
    output_root = default_output_root(input_path, scene_root, args.output_root)

    config, pipeline, checkpoint_path, step = eval_setup(
        config_path,
        eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
        test_mode="test",
        update_config_callback=configure_eval_dataset,
    )

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Scene root: {scene_root}")
    print(f"Output root: {output_root}")
    print(f"Loaded step: {step}")
    print(f"Output mode: {args.output_mode}")

    total_written = 0
    for split in args.split.split("+"):
        split_out_dir = output_root / split
        print(f"Rendering split '{split}' -> {split_out_dir}")
        written = render_split(
            pipeline=pipeline,
            config=config,
            split=split,
            out_dir=split_out_dir,
            near=args.near,
            far_m=args.far_m,
            output_mode=args.output_mode,
            overwrite=args.overwrite,
        )
        total_written += written
        print(f"  wrote {written} EXR files")

    print(f"Done. Total written: {total_written}")


if __name__ == "__main__":
    main()
