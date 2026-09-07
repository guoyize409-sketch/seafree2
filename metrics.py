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
import csv
import os
import re
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_IMAGE_DIR_NAMES = ("images_wb", "images")


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def _image_files(directory):
    directory = Path(directory)
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name,
    )


def _has_image_files(directory):
    directory = Path(directory)
    return directory.is_dir() and any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in directory.iterdir()
    )


def _parse_csv_list(value):
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_scene_gt_dir(gt_root, scene_name, image_dir_names=DEFAULT_IMAGE_DIR_NAMES):
    """Resolve a scene GT image folder from a shared dataset root.

    Priority:
      1. gt_root itself when it already contains images.
      2. gt_root/<scene>/<images_wb|images>.
      3. any recursive <scene>/<images_wb|images> below gt_root.
    """
    gt_root = Path(gt_root)
    if _has_image_files(gt_root):
        return gt_root

    for image_dir_name in image_dir_names:
        candidate = gt_root / scene_name / image_dir_name
        if _has_image_files(candidate):
            return candidate

    for scene_dir in sorted(path for path in gt_root.rglob(scene_name) if path.is_dir()):
        for image_dir_name in image_dir_names:
            candidate = scene_dir / image_dir_name
            if _has_image_files(candidate):
                return candidate

    raise FileNotFoundError(
        f"Could not resolve GT images for scene {scene_name!r} under {gt_root}. "
        f"Tried image dirs: {', '.join(image_dir_names)}"
    )


def list_seafree_scene_dirs(outputs_root, render_subdir="eval_renders"):
    outputs_root = Path(outputs_root)
    return sorted(
        path
        for path in outputs_root.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (path / render_subdir).is_dir()
    )


def _parse_eval_render_index(path, render_prefix):
    pattern = rf"^{re.escape(render_prefix)}_(\d+){re.escape(path.suffix)}$"
    match = re.match(pattern, path.name)
    if match is None:
        raise ValueError(f"Could not parse render index from {path.name} with prefix {render_prefix!r}.")
    return int(match.group(1))


def _list_seafree_renders(renders_dir, render_prefix):
    renders_dir = Path(renders_dir)
    exact_render_pattern = re.compile(rf"^{re.escape(render_prefix)}_(\d+)\.png$")
    renders = sorted(
        [
            path
            for path in renders_dir.glob(f"{render_prefix}_*.png")
            if exact_render_pattern.match(path.name)
        ],
        key=lambda path: _parse_eval_render_index(path, render_prefix),
    )
    if not renders:
        raise FileNotFoundError(
            f"No exact render files matching {render_prefix}_[index].png found in {renders_dir}. "
            f"Diagnostic files such as {render_prefix}_pre_adapter_[index].png are ignored."
        )
    return renders


def _eval_indices_from_interval(num_images, eval_interval):
    if eval_interval <= 0:
        raise ValueError("--eval_interval must be positive.")
    return [idx for idx in range(num_images) if idx % eval_interval == 0]


def _load_metric_tensor(path, device):
    image = Image.open(path).convert("RGB")
    return tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].to(device), image.size


def _load_gt_tensor(gt_path, render_size, device, resize_gt=True):
    gt = Image.open(gt_path).convert("RGB")
    if gt.size != render_size:
        if not resize_gt:
            raise ValueError(
                f"GT/render size mismatch for {gt_path.name}: GT={gt.size}, render={render_size}. "
                "Pass --resize_gt_to_render to resize GT before evaluation."
            )
        gt = gt.resize(render_size, resample=Image.Resampling.BICUBIC)
    return tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].to(device)


def readSeaFreeEvalImages(
    scene_dir,
    gt_dir,
    renders_dir=None,
    render_prefix="eval_rendered_underwater_image",
    eval_interval=8,
    eval_indices=None,
    resize_gt_to_render=True,
    device="cuda:0",
):
    scene_dir = Path(scene_dir)
    renders_dir = Path(renders_dir) if renders_dir is not None else scene_dir / "eval_renders"
    gt_dir = Path(gt_dir)

    render_paths = _list_seafree_renders(renders_dir, render_prefix)
    gt_paths = _image_files(gt_dir)
    if not gt_paths:
        raise FileNotFoundError(f"No GT image files found in {gt_dir}.")

    if eval_indices is None:
        eval_indices = _eval_indices_from_interval(len(gt_paths), eval_interval)
    else:
        eval_indices = [int(idx) for idx in eval_indices]

    renders = []
    gts = []
    image_names = []
    mapping = []
    device = torch.device(device)

    for render_path in render_paths:
        render_idx = _parse_eval_render_index(render_path, render_prefix)
        if render_idx >= len(eval_indices):
            raise IndexError(
                f"Render index {render_idx} from {render_path.name} exceeds eval index list length {len(eval_indices)}."
            )
        gt_idx = eval_indices[render_idx]
        if gt_idx >= len(gt_paths):
            raise IndexError(f"GT index {gt_idx} exceeds GT image count {len(gt_paths)}.")

        render_tensor, render_size = _load_metric_tensor(render_path, device)
        gt_tensor = _load_gt_tensor(gt_paths[gt_idx], render_size, device, resize_gt=resize_gt_to_render)
        renders.append(render_tensor)
        gts.append(gt_tensor)
        image_names.append(gt_paths[gt_idx].name)
        mapping.append(
            {
                "render": render_path.name,
                "render_index": render_idx,
                "gt": gt_paths[gt_idx].name,
                "gt_index": gt_idx,
                "render_size": list(render_size),
            }
        )
    return renders, gts, image_names, mapping


def evaluate_seafree_eval(
    scene_dir,
    gt_dir,
    renders_dir=None,
    render_prefix="eval_rendered_underwater_image",
    eval_interval=8,
    eval_indices=None,
    resize_gt_to_render=True,
    output_json=None,
    device="cuda:0",
):
    print("")
    print("Scene:", scene_dir)
    print("SeaFree eval renders:", renders_dir or str(Path(scene_dir) / "eval_renders"))
    print("GT images:", gt_dir)
    print("Render prefix:", render_prefix)

    renders, gts, image_names, mapping = readSeaFreeEvalImages(
        scene_dir=scene_dir,
        gt_dir=gt_dir,
        renders_dir=renders_dir,
        render_prefix=render_prefix,
        eval_interval=eval_interval,
        eval_indices=eval_indices,
        resize_gt_to_render=resize_gt_to_render,
        device=device,
    )

    print("Render -> GT mapping:")
    for item in mapping:
        print(
            f"  {item['render']} -> {item['gt']} "
            f"(render_idx={item['render_index']}, gt_idx={item['gt_index']}, size={tuple(item['render_size'])})"
        )

    ssims = []
    psnrs = []
    lpipss = []
    with torch.no_grad():
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

    ssims_t = torch.tensor(ssims)
    psnrs_t = torch.tensor(psnrs)
    lpipss_t = torch.tensor(lpipss)

    print("  SSIM : {:>12.7f}".format(ssims_t.mean()))
    print("  PSNR : {:>12.7f}".format(psnrs_t.mean()))
    print("  LPIPS: {:>12.7f}".format(lpipss_t.mean()))
    print("")

    results = {
        "SSIM": ssims_t.mean().item(),
        "PSNR": psnrs_t.mean().item(),
        "LPIPS": lpipss_t.mean().item(),
    }
    per_view = {
        "SSIM": {name: value for name, value in zip(image_names, ssims_t.tolist())},
        "PSNR": {name: value for name, value in zip(image_names, psnrs_t.tolist())},
        "LPIPS": {name: value for name, value in zip(image_names, lpipss_t.tolist())},
    }
    out = {
        "scene": str(scene_dir),
        "renders_dir": str(renders_dir or Path(scene_dir) / "eval_renders"),
        "gt_dir": str(gt_dir),
        "render_prefix": render_prefix,
        "eval_interval": eval_interval,
        "resize_gt_to_render": resize_gt_to_render,
        "mapping": mapping,
        "results": results,
        "per_view": per_view,
    }

    output_json = Path(output_json) if output_json is not None else Path(scene_dir) / f"metrics_{render_prefix}.json"
    with output_json.open("w") as fp:
        json.dump(out, fp, indent=2)
    print(f"Wrote metrics to {output_json}")
    return out


def _write_batch_csv(csv_path, rows):
    fieldnames = [
        "scene",
        "gt_dir",
        "renders_dir",
        "SSIM",
        "PSNR",
        "LPIPS",
        "num_views",
        "output_json",
        "error",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def evaluate_seafree_eval_batch(
    outputs_root,
    gt_root,
    render_prefix="eval_rendered_underwater_image",
    render_subdir="eval_renders",
    eval_interval=8,
    eval_indices=None,
    resize_gt_to_render=True,
    output_json=None,
    output_csv=None,
    scene_names=None,
    image_dir_names=DEFAULT_IMAGE_DIR_NAMES,
    device="cuda:0",
):
    outputs_root = Path(outputs_root)
    gt_root = Path(gt_root)
    scene_name_filter = set(scene_names or [])
    scene_dirs = list_seafree_scene_dirs(outputs_root, render_subdir=render_subdir)
    if scene_name_filter:
        scene_dirs = [scene_dir for scene_dir in scene_dirs if scene_dir.name in scene_name_filter]
    if not scene_dirs:
        raise FileNotFoundError(f"No scene dirs with {render_subdir!r} found under {outputs_root}.")

    rows = []
    scene_results = {}
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        renders_dir = scene_dir / render_subdir
        row = {
            "scene": scene_name,
            "renders_dir": str(renders_dir),
        }
        try:
            gt_dir = resolve_scene_gt_dir(gt_root, scene_name, image_dir_names=image_dir_names)
            scene_output_json = scene_dir / f"metrics_{render_prefix}.json"
            result = evaluate_seafree_eval(
                scene_dir=scene_dir,
                gt_dir=gt_dir,
                renders_dir=renders_dir,
                render_prefix=render_prefix,
                eval_interval=eval_interval,
                eval_indices=eval_indices,
                resize_gt_to_render=resize_gt_to_render,
                output_json=scene_output_json,
                device=device,
            )
            scene_results[scene_name] = result
            row.update(
                {
                    "gt_dir": str(gt_dir),
                    "SSIM": result["results"]["SSIM"],
                    "PSNR": result["results"]["PSNR"],
                    "LPIPS": result["results"]["LPIPS"],
                    "num_views": len(result["mapping"]),
                    "output_json": str(scene_output_json),
                }
            )
        except Exception as exc:
            row["error"] = str(exc)
            print(f"[ERROR] {scene_name}: {exc}")
        rows.append(row)

    successful_rows = [row for row in rows if not row.get("error")]
    mean = {}
    for key in ("SSIM", "PSNR", "LPIPS"):
        values = [float(row[key]) for row in successful_rows if row.get(key) not in (None, "")]
        mean[key] = sum(values) / len(values) if values else None

    summary = {
        "outputs_root": str(outputs_root),
        "gt_root": str(gt_root),
        "render_prefix": render_prefix,
        "render_subdir": render_subdir,
        "eval_interval": eval_interval,
        "resize_gt_to_render": resize_gt_to_render,
        "scene_count": len(rows),
        "successful_scene_count": len(successful_rows),
        "mean_of_scene_means": mean,
        "rows": rows,
    }

    output_json = Path(output_json) if output_json is not None else outputs_root / f"metrics_{render_prefix}_all.json"
    output_csv = Path(output_csv) if output_csv is not None else outputs_root / f"metrics_{render_prefix}_all.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    _write_batch_csv(output_csv, rows)
    print(f"Wrote batch metrics to {output_json}")
    print(f"Wrote batch CSV to {output_csv}")
    return summary


def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=False, nargs="+", type=str, default=[])
    parser.add_argument("--seafree_eval", action="store_true", help="Evaluate SeaFree-GS eval_renders output.")
    parser.add_argument("--seafree_eval_all", action="store_true", help="Evaluate all SeaFree-GS scene folders under --outputs_root.")
    parser.add_argument("--outputs_root", type=str, default=None, help="Root containing per-scene SeaFree-GS output folders.")
    parser.add_argument("--gt_root", type=str, default=None, help="Dataset root used to resolve per-scene GT image folders.")
    parser.add_argument("--gt_dir", type=str, default=None, help="Ground-truth image directory for --seafree_eval.")
    parser.add_argument("--renders_dir", type=str, default=None, help="Render directory. Defaults to <model_path>/eval_renders.")
    parser.add_argument("--render_subdir", type=str, default="eval_renders", help="Per-scene render subdirectory for --seafree_eval_all.")
    parser.add_argument("--render_prefix", type=str, default="eval_rendered_underwater_image", help="SeaFree render filename prefix.")
    parser.add_argument("--eval_interval", type=int, default=8, help="Nerfstudio interval split used for eval images.")
    parser.add_argument("--scene_names", type=str, default=None, help="Optional comma-separated scene names for --seafree_eval_all.")
    parser.add_argument(
        "--image_dir_names",
        type=str,
        default="images_wb,images",
        help="Comma-separated GT image folder names to try under each scene.",
    )
    parser.add_argument(
        "--eval_indices",
        type=str,
        default=None,
        help="Optional comma-separated original GT indices, overriding --eval_interval.",
    )
    parser.add_argument(
        "--resize_gt_to_render",
        action="store_true",
        default=True,
        help="Resize GT images to render resolution before computing metrics.",
    )
    parser.add_argument(
        "--no_resize_gt_to_render",
        action="store_false",
        dest="resize_gt_to_render",
        help="Disable GT resizing and fail on shape mismatch.",
    )
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path for --seafree_eval.")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path for --seafree_eval_all.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    if args.seafree_eval_all:
        if args.outputs_root is None:
            raise ValueError("--outputs_root is required with --seafree_eval_all.")
        if args.gt_root is None:
            raise ValueError("--gt_root is required with --seafree_eval_all.")
        eval_indices = None
        if args.eval_indices is not None:
            eval_indices = [int(part) for part in args.eval_indices.split(",") if part.strip()]
        evaluate_seafree_eval_batch(
            outputs_root=args.outputs_root,
            gt_root=args.gt_root,
            render_prefix=args.render_prefix,
            render_subdir=args.render_subdir,
            eval_interval=args.eval_interval,
            eval_indices=eval_indices,
            resize_gt_to_render=args.resize_gt_to_render,
            output_json=args.output_json,
            output_csv=args.output_csv,
            scene_names=_parse_csv_list(args.scene_names),
            image_dir_names=tuple(_parse_csv_list(args.image_dir_names)),
            device=args.device,
        )
    elif args.seafree_eval:
        if len(args.model_paths) != 1:
            raise ValueError("--seafree_eval expects exactly one --model_paths value.")
        if args.gt_dir is None:
            raise ValueError("--gt_dir is required with --seafree_eval.")
        eval_indices = None
        if args.eval_indices is not None:
            eval_indices = [int(part) for part in args.eval_indices.split(",") if part.strip()]
        evaluate_seafree_eval(
            scene_dir=args.model_paths[0],
            gt_dir=args.gt_dir,
            renders_dir=args.renders_dir,
            render_prefix=args.render_prefix,
            eval_interval=args.eval_interval,
            eval_indices=eval_indices,
            resize_gt_to_render=args.resize_gt_to_render,
            output_json=args.output_json,
            device=args.device,
        )
    else:
        if not args.model_paths:
            raise ValueError("--model_paths is required unless --seafree_eval or --seafree_eval_all is set.")
        evaluate(args.model_paths)
