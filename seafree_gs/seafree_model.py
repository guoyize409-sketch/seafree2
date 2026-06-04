# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SeaFree-GS model implementation based on the Nerfstudio Splatfacto structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

import cv2
import numpy as np

from gsplat.strategy import DefaultStrategy

try:
    from gsplat.rendering import rasterization
except ImportError as exc:
    raise ImportError("SeaFree-GS requires the bundled gsplat package. Install it with `pip install -e third_party/gsplat`.") from exc
        
from pytorch_msssim import SSIM
from torch.nn import Parameter


from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import SHEncoding

from seafree_gs.math import k_nearest_sklearn, random_quat_tensor
from seafree_gs.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases

from torchmetrics.functional.regression import pearson_corrcoef


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class SeaFreeGsModelConfig(ModelConfig):
    """Configuration for the SeaFree-GS model."""

    _target: Type = field(default_factory=lambda: SeaFreeGsModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""


    reset_alpha_value: float = -1.0
    cull_alpha_thresh_post: float = -1.0
    continue_cull_post_densification: bool = False

    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    use_absgrad: bool = True
    """Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad"""

    densify_size_thresh: float = 0.001
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    use_bilateral_grid: bool = False
    """If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/)."""
    grid_shape: Tuple[int, int, int] = (16, 16, 8)
    """Shape of the bilateral grid (X, Y, W)"""
    color_corrected_metrics: bool = False
    """If True, apply color correction to the rendered images before computing the metrics."""

    num_layers_wpp: int = 2
    """Number of hidden layers for the Water Properties Predictor."""
    
    hidden_dim_wpp: int = 128
    """Dimension of hidden layers for the Water Properties Predictor."""
    
    mlp_type: Literal["tcnn", "torch"] = "tcnn"
    """Type of MLP to use for the Water Properties Predictor."""

    enable_coarse_grained_depth_loss: bool = True
        
    enable_background_water_supervision: bool = True
    
class SeaFreeGsModel(Model):
    """SeaFree-GS model for underwater 3D Gaussian Splatting.

    Args:
        config: SeaFree-GS configuration to instantiate the model.
    """

    config: SeaFreeGsModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_depth: float = -1.0,
        **kwargs,
    ):
        self.seed_points = seed_points
        self.max_depth = max_depth * 4
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        

        
        self.line_of_sight_direction_encoding = SHEncoding(levels=4, implementation="tcnn")
        self.ambient_light_activation = nn.Sigmoid()
        self.water_coefficient_activation = nn.Softplus()
        num_layers_wpp = self.config.num_layers_wpp
        hidden_dim_wpp = self.config.hidden_dim_wpp
        mlp_dim_in = self.line_of_sight_direction_encoding.get_out_dim()            
        
        if num_layers_wpp > 1:
            self.water_properties_predictor = MLP(
                in_dim=mlp_dim_in,
                num_layers=num_layers_wpp,
                layer_width=hidden_dim_wpp,
                out_dim=9,
                activation=nn.Sigmoid(),
                out_activation=None,
                implementation=self.config.mlp_type,
            )
        else:
            self.water_properties_predictor = nn.Linear(mlp_dim_in, 9)
            self.config.mlp_type = "torch"
        
            
        
        
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        distances, _ = k_nearest_sklearn(means.data, 3)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)
        
        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

        # Strategy for GS densification
        self.strategy = DefaultStrategy(
            prune_opa=self.config.cull_alpha_thresh,
            grow_grad2d=self.config.densify_grad_thresh,
            grow_scale3d=self.config.densify_size_thresh,
            grow_scale2d=self.config.split_screen_size,
            prune_scale3d=self.config.cull_scale_thresh,
            prune_scale2d=self.config.cull_screen_size,
            refine_scale2d_stop_iter=self.config.stop_screen_size_at,
            refine_start_iter=self.config.warmup_length,
            refine_stop_iter=self.config.stop_split_at,
            reset_every=self.config.reset_alpha_every * self.config.refine_every,
            refine_every=self.config.refine_every,
            pause_refine_after_reset=self.num_train_data + self.config.refine_every,
            absgrad=self.config.use_absgrad,
            revised_opacity=False,
            verbose=True,
            
            reset_alpha_value=self.config.reset_alpha_value,
            cull_alpha_thresh_post=self.config.cull_alpha_thresh_post,
            continue_cull_post_densification=self.config.continue_cull_post_densification,
        )
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        
        
        self.foreground_mask_cache = {}

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        if self.config.sh_degree > 0:
            return self.features_dc
        else:
            return RGB2SH(torch.sigmoid(self.features_dc))

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def step_post_backward(self, step):
        assert step == self.step
        self.strategy.step_post_backward(
            params=self.gauss_params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.step,
            info=self.info,
            packed=False,
        )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backward,
            )
        )
        return cbs

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        
        if self.config.use_bilateral_grid:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())

        gps["water_properties_predictor"] = list(self.water_properties_predictor.parameters())
        gps["line_of_sight_direction_encoding"] = list(self.line_of_sight_direction_encoding.parameters())
                    
        self.camera_optimizer.get_param_groups(param_groups=gps)
        
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        water_background_image = rgb.clone()
        water_coefficients = background.new_zeros(*rgb.shape[:2], 3)
        return {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
            "background": background,
            "intrinsic_color_render": rgb.clone(),
            "water_background_image": water_background_image,
            "background_backscatter_coefficients": water_coefficients.clone(),
            "background_attenuation_coefficients": water_coefficients.clone(),
        }

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def _apply_bilateral_grid(self, rgb: torch.Tensor, cam_idx: int, H: int, W: int) -> torch.Tensor:
        # make xy grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1.0, H, device=self.device),
            torch.linspace(0, 1.0, W, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        out = slice(
            bil_grids=self.bil_grids,
            rgb=rgb,
            xy=grid_xy,
            grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
        )
        return out["rgb"]

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None
        
        # Compute Gaussian colors after applying the water degradation model.
        camtoworlds = torch.inverse(viewmat)  # [C, 4, 4]
        gaussian_line_of_sight_vectors = means_crop[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
        gaussian_line_of_sight_vectors = gaussian_line_of_sight_vectors.detach()
        gaussian_line_of_sight_vectors = gaussian_line_of_sight_vectors[0, :, :]  # [C, N, 3] -> [N, 3]
        epsilon = 1e-8
        gaussian_line_of_sight_distances = torch.norm(gaussian_line_of_sight_vectors, dim=1, keepdim=True)  # [N, 1]
        gaussian_line_of_sight_directions = gaussian_line_of_sight_vectors / (gaussian_line_of_sight_distances + epsilon)  # [N, 3]
        gaussian_count = gaussian_line_of_sight_directions.shape[0]

        # Query the Water Properties Predictor for Gaussian and pixel LOS directions.
        y = torch.linspace(0.0, H, H, device=self.device)
        x = torch.linspace(0.0, W, W, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        cx = K[0, 0, 2]
        cy = K[0, 1, 2]
        fx = K[0, 0, 0]
        fy = K[0, 1, 1]
        yy = (yy - cy) / fy
        xx = (xx - cx) / fx
        pixel_line_of_sight_directions = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        pixel_line_of_sight_norms = torch.linalg.norm(pixel_line_of_sight_directions, dim=-1, keepdim=True)
        pixel_line_of_sight_directions = pixel_line_of_sight_directions / pixel_line_of_sight_norms
        R = camtoworlds[0, :3, :3]
        pixel_line_of_sight_directions = pixel_line_of_sight_directions @ R.T
        image_shape = pixel_line_of_sight_directions.shape[:-1]
        pixel_line_of_sight_directions_flat = pixel_line_of_sight_directions.view(-1, 3)

        directions_concatenated = torch.cat((gaussian_line_of_sight_directions, pixel_line_of_sight_directions_flat), dim=0)
        encoded_line_of_sight_directions = self.line_of_sight_direction_encoding(directions_concatenated)

        if self.config.mlp_type == "tcnn":
            water_properties_raw = self.water_properties_predictor(encoded_line_of_sight_directions)
        else:
            water_properties_raw = self.water_properties_predictor(encoded_line_of_sight_directions.float())


        # The Water Properties Predictor outputs ambient light A, backscatter beta^B,
        # and attenuation beta^D for each queried LOS direction.
        ambient_light_colors = self.ambient_light_activation(water_properties_raw[..., :3])
        backscatter_coefficients = self.water_coefficient_activation(water_properties_raw[..., 3:6])
        attenuation_coefficients = self.water_coefficient_activation(water_properties_raw[..., 6:9])

        gaussian_ambient_light_colors = ambient_light_colors[:gaussian_count].to(gaussian_line_of_sight_directions)
        gaussian_backscatter_coefficients = backscatter_coefficients[:gaussian_count].to(gaussian_line_of_sight_directions)
        gaussian_attenuation_coefficients = attenuation_coefficients[:gaussian_count].to(gaussian_line_of_sight_directions)

        pixel_ambient_light_colors = ambient_light_colors[gaussian_count:].view(*image_shape, -1).to(gaussian_line_of_sight_directions)
        pixel_backscatter_coefficients = backscatter_coefficients[gaussian_count:].view(*image_shape, -1).to(gaussian_line_of_sight_directions)
        pixel_attenuation_coefficients = attenuation_coefficients[gaussian_count:].view(*image_shape, -1).to(gaussian_line_of_sight_directions)

        normalized_gaussian_line_of_sight_distances = gaussian_line_of_sight_distances / 10  # 10 is depth_scale_factor.
        degraded_gaussian_colors = colors_crop * torch.exp(-gaussian_attenuation_coefficients * normalized_gaussian_line_of_sight_distances) + gaussian_ambient_light_colors * (
            1 - torch.exp(-gaussian_backscatter_coefficients * normalized_gaussian_line_of_sight_distances)
        )
        colors_crop = torch.cat((degraded_gaussian_colors, colors_crop), dim=-1)

        water_background_image = pixel_ambient_light_colors
        background_attenuation_coefficients = pixel_attenuation_coefficients
        background_backscatter_coefficients = pixel_backscatter_coefficients

        # render [C,H,W,D] alpha [1,H,W,1]
        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]
        background = self._get_background_color()

        rgb = render[:, ..., :3] + (1 - alpha) * water_background_image
        rgb = torch.clamp(rgb, 0.0, 1.0)

        intrinsic_color_render = render[:, ..., 3:6]
        intrinsic_color_render_clamped = torch.clamp(intrinsic_color_render, 0.0, 1.0)


        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., -1:]
            valid_depth = depth_im[alpha > 0].detach()
            depth_quantile = valid_depth.quantile(0.95)
            depth_im = torch.where(alpha > 0, depth_im, depth_quantile).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "intrinsic_color_render": intrinsic_color_render_clamped.squeeze(0),  # type: ignore
            "water_background_image": water_background_image,
            "background_backscatter_coefficients": background_backscatter_coefficients,
            "background_attenuation_coefficients": background_attenuation_coefficients,
        }  # type: ignore
            

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        gt_depth = batch["depth_image"]
        gt_depth = gt_depth / gt_depth.max()
        
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        for i in range(3):
            metrics_dict[f"background_attenuation_coefficient_{i}_mean"] = outputs["background_attenuation_coefficients"][:, :, i].mean()
            metrics_dict[f"background_backscatter_coefficient_{i}_mean"] = outputs["background_backscatter_coefficients"][:, :, i].mean()
            metrics_dict[f"water_background_color_{i}_mean"] = outputs["water_background_image"][:, :, i].mean()
            
            metrics_dict[f"background_attenuation_coefficient_{i}_max"] = outputs["background_attenuation_coefficients"][:, :, i].max()
            metrics_dict[f"background_backscatter_coefficient_{i}_max"] = outputs["background_backscatter_coefficients"][:, :, i].max()
            metrics_dict[f"water_background_color_{i}_max"] = outputs["water_background_image"][:, :, i].max()
                
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def split_and_calculate_ssim(self, ssim_func, img1, img2):
        """Split images into four blocks, calculate SSIM per block, and return their mean."""
        H, W = img1.shape[-2], img1.shape[-1]

        h_half, w_half = H // 2, W // 2
        blocks1 = [
            img1[..., :h_half, :w_half],
            img1[..., :h_half, w_half:],
            img1[..., h_half:, :w_half],
            img1[..., h_half:, w_half:],
        ]
        blocks2 = [
            img2[..., :h_half, :w_half],
            img2[..., :h_half, w_half:],
            img2[..., h_half:, :w_half],
            img2[..., h_half:, w_half:]
        ]

        ssim_values = [ssim_func(b1, b2) for b1, b2 in zip(blocks1, blocks2)]
        return torch.mean(torch.stack(ssim_values))
    

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_underwater_image = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        rendered_underwater_image = outputs["rgb"]


        pseudo_depth = self._downscale_if_required(batch["depth_image"])
        pseudo_depth = pseudo_depth.to(self.device)        
        pseudo_depth = pseudo_depth / pseudo_depth.max()
        rendered_depth = outputs["depth"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_underwater_image.shape[:2] == rendered_underwater_image.shape[:2]
            gt_underwater_image = gt_underwater_image * mask
            rendered_underwater_image = rendered_underwater_image * mask
            pseudo_depth = pseudo_depth * mask
            rendered_depth = rendered_depth * mask
            
        # Build the foreground mask from the pseudo-depth map.
        image_idx = batch['image_idx']
        if image_idx in self.foreground_mask_cache.keys() and self._get_downscale_factor() in self.foreground_mask_cache[image_idx].keys():
            foreground_mask = self.foreground_mask_cache[image_idx][self._get_downscale_factor()]['mask_tensor']
            background_pixel_ratio = self.foreground_mask_cache[image_idx][self._get_downscale_factor()]['background_pixel_ratio']

        else:
            background_depth_threshold = 1e-2
            pseudo_depth_numpy = pseudo_depth.squeeze().cpu().numpy()
            mask_1e_2_copy = (pseudo_depth_numpy < background_depth_threshold).astype(np.uint8) * 255
            _, binary_image = cv2.threshold(mask_1e_2_copy, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            foreground_mask_array = np.zeros_like(binary_image)
            if len(contours) > 0:
                max_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(foreground_mask_array, [max_contour], -1, (255,), thickness=cv2.FILLED)
            foreground_mask_array = foreground_mask_array.astype(np.float32) / 255
            foreground_mask_array[foreground_mask_array < background_depth_threshold] = 0
            foreground_mask_array[foreground_mask_array > 0] = 1
            
            foreground_mask = torch.from_numpy(foreground_mask_array).float()
            foreground_mask = foreground_mask.to(device=pseudo_depth.device, dtype=pseudo_depth.dtype)
            foreground_mask.unsqueeze_(-1)
            
            total_pixels = foreground_mask_array.size
            background_pixels = foreground_mask_array[foreground_mask_array < 0.5]
            background_pixel_ratio = background_pixels.size / total_pixels                                
            
            if image_idx not in self.foreground_mask_cache.keys():
                self.foreground_mask_cache[image_idx] = {}
            self.foreground_mask_cache[image_idx][self._get_downscale_factor()] = {
                'mask_tensor': foreground_mask,
                'background_pixel_ratio': background_pixel_ratio,
            }
        
        foreground_aware_reconstruction_weight = 1 / (rendered_underwater_image.detach() + 1e-3)
        foreground_aware_reconstruction_weight = torch.where(
            foreground_mask < 0.5,
            torch.ones_like(foreground_aware_reconstruction_weight),
            foreground_aware_reconstruction_weight,
        )
        foreground_weighted_l1_loss = torch.abs(
            (gt_underwater_image - rendered_underwater_image) * foreground_aware_reconstruction_weight
        ).mean()

        gt_underwater_image_chw = gt_underwater_image.permute(2, 0, 1)[None, ...]
        rendered_underwater_image_chw = rendered_underwater_image.permute(2, 0, 1)[None, ...]

        foreground_aware_reconstruction_weight = foreground_aware_reconstruction_weight.permute(2, 0, 1)[None, ...]
        weighted_gt_underwater_image = gt_underwater_image_chw * foreground_aware_reconstruction_weight
        weighted_rendered_underwater_image = rendered_underwater_image_chw * foreground_aware_reconstruction_weight

        
        if weighted_gt_underwater_image.shape[-1] > 800 or weighted_gt_underwater_image.shape[-2] > 800:
            foreground_weighted_dssim_loss = 1 - self.split_and_calculate_ssim(
                self.ssim, weighted_gt_underwater_image, weighted_rendered_underwater_image
            )
        else:
            foreground_weighted_dssim_loss = 1 - self.ssim(
                weighted_gt_underwater_image, weighted_rendered_underwater_image
            )                
        
        if self.config.use_scale_regularization:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio, device=self.device),
                )
                - self.config.max_gauss_ratio
            )
            
            if self.step < (self.config.reset_alpha_every * self.config.refine_every + 200):
                scale_reg = 0.1 * scale_reg.mean()
            else:
                scale_reg = 100 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        coarse_grained_depth_loss = torch.tensor(0.0, device=self.device)
        pseudo_depth_flattened = pseudo_depth.flatten()
        rendered_depth_flattened = rendered_depth.flatten()
        if self.config.enable_coarse_grained_depth_loss:
            approximate_rendered_disparity = 1 / (rendered_depth_flattened * 10 + 1)
            coarse_grained_depth_loss += 1 - pearson_corrcoef(pseudo_depth_flattened, approximate_rendered_disparity)

        water_background_image = outputs["water_background_image"]
        
        background_water_supervision_loss = torch.tensor(0.0, device=self.device)
        if self.config.enable_background_water_supervision and self.step < 15000:
            if background_pixel_ratio > 0.05:
                background_ambient_light_pixels = water_background_image[foreground_mask.repeat(1, 1, 3) < 0.5]
                background_underwater_pixels = gt_underwater_image[foreground_mask.repeat(1, 1, 3) < 0.5]            
                background_ambient_light_weight = 1 / (background_ambient_light_pixels.detach() + 1e-3)
                background_water_supervision_loss += torch.abs(
                    (background_ambient_light_pixels - background_underwater_pixels) * background_ambient_light_weight
                ).mean()        

        # CB-Loss includes foreground-weighted reconstruction and direct background
        # water supervision as its third term.
        loss_dict = {
            "content_based_reconstruction_loss": (
                (1 - self.config.ssim_lambda) * foreground_weighted_l1_loss
                + self.config.ssim_lambda * foreground_weighted_dssim_loss
                + background_water_supervision_loss * 0.01
            ),
            "scale_regularization_loss": scale_reg,
            "coarse_grained_depth_loss": coarse_grained_depth_loss * 0.1,
        }

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        gt_depth = batch["depth_image"]
        gt_depth = gt_depth / gt_depth.max()
                
        predicted_rgb = outputs["rgb"]
        cc_rgb = None

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        if gt_rgb.shape[-1] > 800 or predicted_rgb.shape[-2] > 800:
            ssim = self.split_and_calculate_ssim(self.ssim, gt_rgb, predicted_rgb)
        else:
            ssim = self.ssim(gt_rgb, predicted_rgb) 
                     
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            if gt_rgb.shape[-1] > 800 or predicted_rgb.shape[-2] > 800:
                cc_ssim = self.split_and_calculate_ssim(self.ssim, gt_rgb, cc_rgb)
            else:
                cc_ssim = self.ssim(gt_rgb, cc_rgb)     
                                      
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        images_dict = {"img": combined_rgb}
        images_dict['accumulation'] = outputs["accumulation"]
        
        images_dict['intrinsic_color_render'] = outputs["intrinsic_color_render"]
        images_dict['rendered_underwater_image'] = outputs["rgb"]  
         
        images_dict['water_background_image'] = outputs["water_background_image"]   
        images_dict['background_backscatter_coefficients'] = outputs["background_backscatter_coefficients"]   
        images_dict['background_attenuation_coefficients'] = outputs["background_attenuation_coefficients"]            
            
        images_dict['depth_GT'] = gt_depth
        
        depth_ED = outputs["depth"]
        q99 = torch.quantile(depth_ED.flatten(), 0.99)
        depth_ED_clipped = torch.clamp(depth_ED, max=q99)
        depth_ED_normalized = depth_ED_clipped / q99
        images_dict['depth_ED'] = depth_ED_normalized

            
        return metrics_dict, images_dict
