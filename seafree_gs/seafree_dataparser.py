"""SeaFree-GS COLMAP dataparser.

This parser inherits the official Nerfstudio COLMAP parser and overrides only
the SeaFree-specific pieces that were validated on the current datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParser, ColmapDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params


@dataclass
class SeaFreeGsDataParserConfig(ColmapDataParserConfig):
    """Configuration for the SeaFree-GS COLMAP dataparser."""

    _target: Type = field(default_factory=lambda: SeaFreeGsDataParser)


class SeaFreeGsDataParser(ColmapDataParser):
    """COLMAP dataparser with SeaFree-specific image ordering and scaling."""

    config: SeaFreeGsDataParserConfig

    def _get_all_images_and_cameras(self, recon_dir: Path):
        if (recon_dir / "cameras.txt").exists():
            cam_id_to_camera = colmap_utils.read_cameras_text(recon_dir / "cameras.txt")
            im_id_to_image = colmap_utils.read_images_text(recon_dir / "images.txt")
        elif (recon_dir / "cameras.bin").exists():
            cam_id_to_camera = colmap_utils.read_cameras_binary(recon_dir / "cameras.bin")
            im_id_to_image = colmap_utils.read_images_binary(recon_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt or cameras.bin in {recon_dir}")

        cameras = {}
        frames = []
        camera_model = None

        for cam_id, cam_data in cam_id_to_camera.items():
            cameras[cam_id] = parse_colmap_camera_params(cam_data)

        # Sort by image filename instead of COLMAP image id for deterministic dataset ordering.
        ordered_im_id = sorted(im_id_to_image.keys(), key=lambda k: im_id_to_image[k].name)

        for im_id in ordered_im_id:
            im_data = im_id_to_image[im_id]
            # NB: COLMAP uses Eigen / scalar-first quaternions.
            rotation = colmap_utils.qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to Nerfstudio's OpenGL convention.
            c2w[0:3, 1:3] *= -1
            if self.config.assume_colmap_world_coordinate_convention:
                # Map COLMAP gravity guess (-y) to Nerfstudio convention (+z).
                c2w = c2w[np.array([0, 2, 1, 3]), :]
                c2w[2, :] *= -1

            frame = {
                "file_path": (self.config.data / self.config.images_path / im_data.name).as_posix(),
                "transform_matrix": c2w,
                "colmap_im_id": im_id,
            }
            frame.update(cameras[im_data.camera_id])
            if self.config.masks_path is not None:
                frame["mask_path"] = (
                    (self.config.data / self.config.masks_path / im_data.name).with_suffix(".png").as_posix()
                )
            if self.config.depths_path is not None:
                frame["depth_path"] = (
                    (self.config.data / self.config.depths_path / im_data.name).with_suffix(".png").as_posix()
                )
            frames.append(frame)
            if camera_model is not None:
                assert camera_model == frame["camera_model"], "Multiple camera models are not supported"
            else:
                camera_model = frame["camera_model"]

        out = {}
        out["frames"] = frames
        if self.config.assume_colmap_world_coordinate_convention:
            # Map COLMAP gravity guess (-y) to Nerfstudio convention (+z).
            applied_transform = np.eye(4)[:3, :]
            applied_transform = applied_transform[np.array([0, 2, 1]), :]
            applied_transform[2, :] *= -1
            out["applied_transform"] = applied_transform.tolist()
        out["camera_model"] = camera_model
        assert len(frames) > 0, "No images found in the colmap model"
        return out

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            if any([k in frame and float(frame[k]) != 0.0 for k in ["k4", "k5", "k6"]]):
                raise ValueError(
                    "K4/K5/K6 is non-zero! Note that Nerfstudio camera model's K4 has different meaning than colmap "
                    "OPENCV camera model K4. Nerfstudio's K4 is the 4-th order of radial distortion coefficient, while "
                    "colmap/OPENCV's K4 is 4-th coefficient in fractional radial distortion model."
                )
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            image_filenames.append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))
            if "depth_path" in frame:
                depth_filenames.append(Path(frame["depth_path"]))

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        indices = self._get_image_indices(image_filenames, split)

        # Resolve image, mask, and depth paths to the active training resolution.
        image_filenames, mask_filenames, depth_filenames, downscale_factor = self._setup_downscale_factor(
            image_filenames, mask_filenames, depth_filenames
        )

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        cameras.rescale_output_resolution(
            scaling_factor=1.0 / downscale_factor,
            scale_rounding_mode=self.config.downscale_rounding_mode,
        )

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        metadata = {}
        if self.config.load_3D_points:
            # Load 3D points
            metadata.update(self._load_3D_points(colmap_path, transform_matrix, scale_factor, downscale_factor))

        bounds_path = colmap_path.parents[1] / "poses_bounds.npy"
        if bounds_path.exists():
            near_far = np.load(bounds_path)[:, -2:]
            max_depth = np.max(near_far[:, 1])
            max_depth *= scale_factor
            metadata["max_depth"] = max_depth

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,
            },
        )
        return dataparser_outputs

    def _load_3D_points(
        self,
        colmap_path: Path,
        transform_matrix: torch.Tensor,
        scale_factor: float,
        downscale_factor: float,
    ):
        if (colmap_path / "points3D.bin").exists():
            colmap_points = colmap_utils.read_points3D_binary(colmap_path / "points3D.bin")
        elif (colmap_path / "points3D.txt").exists():
            colmap_points = colmap_utils.read_points3D_text(colmap_path / "points3D.txt")
        else:
            raise ValueError(f"Could not find points3D.txt or points3D.bin in {colmap_path}")

        points3D = torch.from_numpy(np.array([p.xyz for p in colmap_points.values()], dtype=np.float32))
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor

        # Load point colours
        points3D_rgb = torch.from_numpy(np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8))
        points3D_num_points = torch.tensor([len(p.image_ids) for p in colmap_points.values()], dtype=torch.int64)
        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
            "points3D_error": torch.from_numpy(np.array([p.error for p in colmap_points.values()], dtype=np.float32)),
            "points3D_num_points2D": points3D_num_points,
        }

        if self.config.max_2D_matches_per_3D_point != 0:
            if (colmap_path / "images.txt").exists():
                im_id_to_image = colmap_utils.read_images_text(colmap_path / "images.txt")
            elif (colmap_path / "images.bin").exists():
                im_id_to_image = colmap_utils.read_images_binary(colmap_path / "images.bin")
            else:
                raise ValueError(f"Could not find images.txt or images.bin in {colmap_path}")

            max_num_points = int(torch.max(points3D_num_points).item())
            if self.config.max_2D_matches_per_3D_point > 0:
                max_num_points = min(max_num_points, self.config.max_2D_matches_per_3D_point)
            points3D_image_ids = []
            points3D_image_xy = []
            for p in colmap_points.values():
                nids = np.array(p.image_ids, dtype=np.int64)
                nxy_ids = np.array(p.point2D_idxs, dtype=np.int32)
                if self.config.max_2D_matches_per_3D_point != -1:
                    idxs = np.argsort(p.error)[: self.config.max_2D_matches_per_3D_point]
                    nids = nids[idxs]
                    nxy_ids = nxy_ids[idxs]
                nxy = [im_id_to_image[im_id].xys[pt_idx] for im_id, pt_idx in zip(nids, nxy_ids)]
                nxy = torch.from_numpy(np.stack(nxy).astype(np.float32))
                nids = torch.from_numpy(nids)
                assert len(nids.shape) == 1
                assert len(nxy.shape) == 2
                points3D_image_ids.append(
                    torch.cat((nids, torch.full((max_num_points - len(nids),), -1, dtype=torch.int64)))
                )
                points3D_image_xy.append(
                    torch.cat((nxy, torch.full((max_num_points - len(nxy), nxy.shape[-1]), 0, dtype=torch.float32)))
                    / downscale_factor
                )
            out["points3D_image_ids"] = torch.stack(points3D_image_ids, dim=0)
            out["points3D_points2D_xy"] = torch.stack(points3D_image_xy, dim=0)
        return out

