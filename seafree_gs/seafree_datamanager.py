"""SeaFree-GS full-image datamanager."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Literal, Type

import torch
from rich.progress import track
from typing_extensions import assert_never

from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
    _undistort_image,
)
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SeaFreeGsFullImageDatamanagerConfig(FullImageDatamanagerConfig):
    """Configuration for the SeaFree-GS full-image datamanager."""

    _target: Type = field(default_factory=lambda: SeaFreeGsFullImageDatamanager)


class SeaFreeGsFullImageDatamanager(FullImageDatamanager, Generic[TDataset]):
    """Full-image datamanager that keeps depth images on GPU when image caching uses GPU."""

    config: SeaFreeGsFullImageDatamanagerConfig

    @property
    def dataset_type(self) -> Type[DepthDataset]:
        return DepthDataset

    def _load_images(
        self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
    ) -> List[Dict[str, torch.Tensor]]:
        undistorted_images: List[Dict[str, torch.Tensor]] = []

        if split == "train":
            dataset = self.train_dataset
        elif split == "eval":
            dataset = self.eval_dataset
        else:
            assert_never(split)

        def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
            data = dataset.get_data(idx, image_type=self.config.cache_images_type)
            camera = dataset.cameras[idx].reshape(())
            assert data["image"].shape[1] == camera.width.item() and data["image"].shape[0] == camera.height.item(), (
                f'The size of image ({data["image"].shape[1]}, {data["image"].shape[0]}) loaded '
                f"does not match the camera parameters ({camera.width.item(), camera.height.item()})"
            )
            if camera.distortion_params is None or torch.all(camera.distortion_params == 0):
                return data
            K = camera.get_intrinsics_matrices().numpy()
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
            data["image"] = torch.from_numpy(image)
            if mask is not None:
                data["mask"] = mask

            dataset.cameras.fx[idx] = float(K[0, 0])
            dataset.cameras.fy[idx] = float(K[1, 1])
            dataset.cameras.cx[idx] = float(K[0, 2])
            dataset.cameras.cy[idx] = float(K[1, 2])
            dataset.cameras.width[idx] = image.shape[1]
            dataset.cameras.height[idx] = image.shape[0]
            return data

        CONSOLE.log(f"Caching / undistorting {split} images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            undistorted_images = list(
                track(
                    executor.map(
                        undistort_idx,
                        range(len(dataset)),
                    ),
                    description=f"Caching / undistorting {split} images",
                    transient=True,
                    total=len(dataset),
                )
            )

        if cache_images_device == "gpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
                if "depth" in cache:
                    cache["depth"] = cache["depth"].to(self.device)
                if "depth_image" in cache:
                    cache["depth_image"] = cache["depth_image"].to(self.device)
                self.train_cameras = self.train_dataset.cameras.to(self.device)
        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
                self.train_cameras = self.train_dataset.cameras
        else:
            assert_never(cache_images_device)

        return undistorted_images
