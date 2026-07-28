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
import torch.nn.functional as F

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

    enable_water_null_occupancy: bool = True
    """Enable water-null occupancy suppression for pure-water background rays."""
    water_null_start_step: int = 500
    """Start applying water-null losses after this training step."""
    water_null_depth_threshold: float = 1e-2
    """Minimum normalized pseudo-depth threshold used to identify far / water-only rays."""
    water_null_depth_quantile: float = 0.04
    """Adaptive low-depth quantile used for pure-water probability estimation."""
    water_transition_depth_quantile: float = 0.22
    """Adaptive mid-low pseudo-depth quantile used for transition-water estimation."""
    water_null_depth_sharpness: float = 12.0
    """Sigmoid sharpness for depth-based water probability."""
    water_null_texture_quantile: float = 0.20
    """Image-gradient quantile below which pixels are considered water-like and low texture."""
    water_null_texture_sharpness: float = 10.0
    """Sigmoid sharpness for texture-based water probability."""
    water_transition_intrinsic_texture_quantile: float = 0.25
    """Intrinsic-render gradient quantile below which transition water remains texture-poor."""
    enable_intrinsic_scene_rescue: bool = True
    """Use raw intrinsic appearance as counter-evidence before suppressing water/null regions."""
    intrinsic_rescue_luma_threshold: float = 0.035
    """Minimum raw intrinsic luminance required before color evidence can rescue a region."""
    intrinsic_rescue_texture_quantile: float = 0.65
    """Raw intrinsic gradient quantile above which local structure becomes scene evidence."""
    intrinsic_rescue_color_sigma: float = 0.18
    """Color-distance scale for raw intrinsic vs. water-background scene evidence."""
    intrinsic_rescue_alpha_threshold: float = 0.20
    """Gaussian alpha threshold used as supporting evidence for weak intrinsic scene structure."""
    intrinsic_rescue_pure_strength: float = 0.85
    """How strongly scene evidence suppresses pure-water probability."""
    intrinsic_rescue_transition_strength: float = 0.95
    """How strongly scene evidence suppresses transition-water probability."""
    intrinsic_visibility_rescue_strength: float = 0.90
    """How strongly scene evidence protects the final intrinsic visibility gate."""
    intrinsic_visibility_floor: float = 0.35
    """Minimum visibility retained when raw intrinsic scene evidence is strong."""
    intrinsic_rescue_structure_strength: float = 0.35
    """Maximum amount by which texture/depth structure can amplify appearance-backed scene rescue."""
    intrinsic_rescue_blue_suppression: float = 0.90
    """Suppress scene rescue in raw-intrinsic blue-water chroma regions."""
    intrinsic_water_blue_margin: float = 0.035
    """Blue-channel excess above max(red, green) that indicates residual water color."""
    intrinsic_water_blue_scale: float = 0.035
    """Softness of the raw-intrinsic blue-water chroma detector."""
    intrinsic_water_red_deficit_margin: float = 0.030
    """Red-channel deficit below green/blue that indicates cyan water residuals."""
    intrinsic_water_red_deficit_scale: float = 0.040
    """Softness of the raw-intrinsic cyan / red-deficit water detector."""
    residual_water_strength: float = 1.00
    """Strength with which residual blue-water evidence contributes to null probability."""
    residual_water_visibility_strength: float = 1.00
    """Strength with which residual blue-water evidence suppresses final intrinsic visibility."""
    residual_water_visibility_gamma: float = 1.80
    """Nonlinear visibility hardening for residual water; >1 makes medium-confidence water darker."""
    residual_water_scene_suppression: float = 0.90
    """How strongly residual blue-water evidence cancels scene rescue in visibility."""
    residual_water_depth_floor: float = 0.65
    """Minimum depth-context confidence retained for raw-intrinsic residual water evidence."""
    residual_water_similarity_floor: float = 0.55
    """Minimum water-color similarity retained for raw-intrinsic residual water evidence."""
    residual_water_scene_guard_floor: float = 0.50
    """Minimum residual-water confidence retained even when scene rescue is non-zero."""
    residual_water_chroma_loss_weight: float = 0.01
    """Penalty weight for blue-channel residuals in raw intrinsic water-color regions."""
    enable_connected_freewater_prior: bool = True
    """Use soft border-connected water evidence to suppress free-water sheets and bands."""
    connected_freewater_iterations: int = 24
    """Number of 3x3 soft dilation steps for the border-connected freewater prior."""
    connected_freewater_strength: float = 0.85
    """Strength with which connected freewater evidence contributes to null probability."""
    connected_freewater_visibility_strength: float = 1.00
    """Strength with which connected freewater evidence suppresses final intrinsic visibility."""
    connected_freewater_visibility_gamma: float = 1.40
    """Nonlinear visibility hardening for border-connected freewater sheets."""
    water_null_ambient_color_sigma: float = 0.10
    """Color-distance scale for matching pure-water pixels to WPP ambient light."""
    water_transition_ambient_color_sigma: float = 0.16
    """Looser ambient-color scale used for blue-gray transition water."""
    water_depth_edge_quantile: float = 0.65
    """Depth-gradient quantile above which pixels are protected as geometric boundaries."""
    water_depth_edge_sharpness: float = 8.0
    """Sigmoid sharpness for depth-edge transition confidence."""
    water_null_edge_quantile: float = 0.70
    """Image-gradient quantile above which pixels are protected as scene structure."""
    water_null_edge_sharpness: float = 8.0
    """Sigmoid sharpness for edge/texture-based scene protection."""
    water_null_edge_protection: float = 0.85
    """Maximum down-weight applied to likely scene edges in the water probability."""
    enable_water_horizon_prior: bool = True
    """Enable per-image free-water horizon / boundary prior."""
    water_horizon_min_confidence: float = 0.10
    """Minimum top-band water confidence required before horizon prior is active."""
    water_horizon_strength: float = 0.45
    """Strength of the free-water horizon prior added to transition-water probability."""
    water_horizon_band: float = 0.12
    """Soft transition band width for the normalized horizon boundary."""
    water_null_binary_threshold: float = 0.50
    """Threshold used when converting soft water probability to foreground/background masks."""
    water_null_strict_threshold: float = 0.70
    """Minimum soft water probability required before destructive water-null losses are active."""
    water_pure_max_fraction: float = 0.04
    """Maximum per-image pixel fraction allowed to receive pure-water destructive losses."""
    water_transition_strict_threshold: float = 0.45
    """Minimum transition probability required before transition losses are active."""
    water_transition_max_fraction: float = 0.18
    """Maximum per-image pixel fraction allowed to receive transition-water losses."""
    water_transition_alpha_target: float = 0.15
    """Allowed residual Gaussian alpha in transition water before alpha loss is applied."""
    water_empty_loss_weight: float = 0.08
    """Penalty weight for Gaussian accumulation on pure-water rays."""
    water_transition_alpha_loss_weight: float = 0.0
    """Penalty weight for keeping transition-water alpha below its low target. Kept off by default because transition masks are intentionally soft."""
    water_intrinsic_null_loss_weight: float = 0.02
    """Penalty weight for non-black intrinsic radiance on pure-water rays."""
    water_background_ownership_loss_weight: float = 0.02
    """Penalty weight that makes WPP, not scene Gaussians, explain pure-water observations."""
    water_lifecycle_ema: float = 0.96
    """EMA factor for per-Gaussian water/scene exposure scores."""
    enable_water_lifecycle_growth_suppression: bool = False
    """Allow water-dominated Gaussians to be blocked from densification. Disabled by default to avoid suppressing weak distant objects."""
    enable_water_lifecycle_pruning: bool = False
    """Allow water-dominated Gaussians to be pruned. Disabled by default because false positives permanently remove scene detail."""
    water_growth_suppress_start_step: int = 6000
    """Start suppressing densification gradients for water-dominated Gaussians."""
    water_growth_suppress_threshold: float = 0.75
    """Water exposure score above which Gaussian densification is suppressed."""
    water_growth_suppress_ratio: float = 2.5
    """Required water-vs-scene exposure ratio for densification suppression."""
    water_growth_grad_scale: float = 0.35
    """Scale applied to 2D gradient statistics for water-dominated Gaussians."""
    water_depth_edge_support_max: float = 0.22
    """Maximum cross-view depth-edge support for a Gaussian to be treated as null-dominated."""
    water_color_residual_support_max: float = 0.30
    """Maximum cross-view color-residual support for a Gaussian to be treated as null-dominated."""
    water_view_consistency_min: float = 0.25
    """Minimum decisive cross-view evidence required for null growth blocking / pruning."""
    water_prune_start_step: int = 15000
    """Start pruning water-dominated Gaussians after this training step."""
    water_prune_score_threshold: float = 0.85
    """Water exposure score above which Gaussians become pruning candidates."""
    water_prune_scene_score_max: float = 0.20
    """Maximum scene exposure score for a water-dominated Gaussian to be pruned."""
    water_prune_score_ratio: float = 3.0
    """Required water-vs-scene exposure ratio for water-aware pruning."""
    water_prune_max_fraction: float = 0.005
    """Maximum fraction of current Gaussians pruned by the water-aware pass at one refine step."""
    
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
        self.strategy_state["water_exposure_score"] = torch.zeros(num_points, device=means.device)
        self.strategy_state["scene_exposure_score"] = torch.zeros(num_points, device=means.device)
        self.strategy_state["depth_edge_support_score"] = torch.zeros(num_points, device=means.device)
        self.strategy_state["color_residual_support_score"] = torch.zeros(num_points, device=means.device)
        self.strategy_state["view_consistency_score"] = torch.zeros(num_points, device=means.device)
        
        self.foreground_mask_cache = {}
        self.register_buffer("water_exposure_score", torch.zeros(num_points), persistent=False)
        self.register_buffer("scene_exposure_score", torch.zeros(num_points), persistent=False)
        self.register_buffer("depth_edge_support_score", torch.zeros(num_points), persistent=False)
        self.register_buffer("color_residual_support_score", torch.zeros(num_points), persistent=False)
        self.register_buffer("view_consistency_score", torch.zeros(num_points), persistent=False)
        self.last_water_probability_mean = torch.tensor(0.0)
        self.last_water_pure_probability_mean = torch.tensor(0.0)
        self.last_water_transition_probability_mean = torch.tensor(0.0)
        self.last_water_strict_probability_mean = torch.tensor(0.0)
        self.last_water_horizon_probability_mean = torch.tensor(0.0)
        self.last_residual_water_probability_mean = torch.tensor(0.0)
        self.last_connected_freewater_probability_mean = torch.tensor(0.0)
        self.last_water_no_grow_candidates = 0
        self.last_water_prune_candidates = 0

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
        if self.config.enable_water_null_occupancy:
            if self.config.enable_water_lifecycle_growth_suppression:
                self._suppress_water_dominated_growth()
                water_no_grow_mask = self._get_water_lifecycle_no_grow_mask()
                if hasattr(self.strategy, "set_external_no_grow_mask"):
                    self.strategy.set_external_no_grow_mask(water_no_grow_mask)
                self.last_water_no_grow_candidates = int(water_no_grow_mask.sum().item()) if water_no_grow_mask is not None else 0
            else:
                self.last_water_no_grow_candidates = 0
            if self.config.enable_water_lifecycle_pruning:
                water_prune_mask = self._get_water_lifecycle_prune_mask()
                if hasattr(self.strategy, "set_external_prune_mask"):
                    self.strategy.set_external_prune_mask(water_prune_mask)
                self.last_water_prune_candidates = int(water_prune_mask.sum().item()) if water_prune_mask is not None else 0
            else:
                self.last_water_prune_candidates = 0
        n_before_refine = self.num_points
        self.strategy.step_post_backward(
            params=self.gauss_params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.step,
            info=self.info,
            packed=False,
        )
        if self.config.enable_water_null_occupancy:
            self._sync_water_lifecycle_buffers(n_before_refine)

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
    def _weighted_mean(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute a stable weighted mean for soft water/null masks."""
        expanded_weight = weight
        while expanded_weight.dim() < value.dim():
            expanded_weight = expanded_weight.unsqueeze(-1)
        if expanded_weight.shape != value.shape:
            expanded_weight = expanded_weight.expand_as(value)
        return (value * expanded_weight).sum() / expanded_weight.sum().clamp_min(eps)

    def _image_gradient_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        """Low-cost image-gradient magnitude used for water-like smooth-region confidence."""
        gray = image[..., :3].mean(dim=-1, keepdim=True)
        dx = torch.zeros_like(gray)
        dy = torch.zeros_like(gray)
        dx[:, 1:, :] = torch.abs(gray[:, 1:, :] - gray[:, :-1, :])
        dy[1:, :, :] = torch.abs(gray[1:, :, :] - gray[:-1, :, :])
        return dx + dy

    def _intrinsic_blue_water_chroma(self, intrinsic_image: Optional[torch.Tensor]) -> torch.Tensor:
        """Detect visible blue/cyan residual water color in the raw intrinsic render."""
        if intrinsic_image is None or intrinsic_image.numel() == 0:
            raise ValueError("intrinsic_image is required to infer the output shape")
        intrinsic = torch.nan_to_num(
            intrinsic_image[..., :3].detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        red_or_green = torch.maximum(intrinsic[..., 0:1], intrinsic[..., 1:2])
        blue = intrinsic[..., 2:3]
        blue_excess = blue - red_or_green
        blue_margin = self.config.intrinsic_water_blue_margin
        blue_scale = max(self.config.intrinsic_water_blue_scale, 1e-4)
        blue_chroma = torch.sigmoid((blue_excess - blue_margin) / blue_scale * 4.0)
        red_deficit = torch.minimum(intrinsic[..., 1:2], blue) - intrinsic[..., 0:1]
        red_deficit_margin = self.config.intrinsic_water_red_deficit_margin
        red_deficit_scale = max(self.config.intrinsic_water_red_deficit_scale, 1e-4)
        cyan_chroma = torch.sigmoid(
            (red_deficit - red_deficit_margin) / red_deficit_scale * 4.0
        )

        luma = intrinsic.mean(dim=-1, keepdim=True)
        luma_threshold = max(self.config.intrinsic_rescue_luma_threshold, 1e-4)
        visible_luma = torch.sigmoid((luma - luma_threshold) / luma_threshold * 4.0)
        return (torch.maximum(blue_chroma, 0.75 * cyan_chroma) * visible_luma).detach().clamp(0.0, 1.0)

    def _build_connected_freewater_prior(
        self,
        seed: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        """Softly propagate water evidence from image borders through water-like support."""
        if not self.config.enable_connected_freewater_prior or seed.numel() == 0:
            return torch.zeros_like(seed)

        seed_2d = seed.detach().squeeze(-1).clamp(0.0, 1.0)
        support_2d = support.detach().squeeze(-1).clamp(0.0, 1.0)
        border = torch.zeros_like(seed_2d)
        border[0, :] = seed_2d[0, :]
        border[-1, :] = torch.maximum(border[-1, :], seed_2d[-1, :])
        border[:, 0] = torch.maximum(border[:, 0], seed_2d[:, 0])
        border[:, -1] = torch.maximum(border[:, -1], seed_2d[:, -1])

        connected = border[None, None, ...]
        support_chw = support_2d[None, None, ...]
        for _ in range(max(0, int(self.config.connected_freewater_iterations))):
            expanded = F.max_pool2d(connected, kernel_size=3, stride=1, padding=1)
            connected = torch.maximum(connected, expanded * support_chw)
        return connected.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)

    def _build_intrinsic_scene_rescue(
        self,
        intrinsic_image: Optional[torch.Tensor],
        water_background_image: torch.Tensor,
        intrinsic_texture: Optional[torch.Tensor] = None,
        depth_edge_support: Optional[torch.Tensor] = None,
        accumulation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build raw-intrinsic counter-evidence that a blue / distant region is still real scene content."""
        reference = water_background_image[..., :1]
        if (
            not self.config.enable_intrinsic_scene_rescue
            or intrinsic_image is None
            or intrinsic_image.numel() == 0
        ):
            return torch.zeros_like(reference)

        intrinsic = torch.nan_to_num(
            intrinsic_image[..., :3].detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        intrinsic_luma = intrinsic.mean(dim=-1, keepdim=True)
        luma_threshold = max(self.config.intrinsic_rescue_luma_threshold, 1e-4)
        luma_evidence = torch.sigmoid((intrinsic_luma - luma_threshold) / luma_threshold * 6.0)

        if intrinsic_texture is None:
            intrinsic_texture = self._image_gradient_magnitude(intrinsic).detach()
        texture_cut = torch.quantile(
            intrinsic_texture.flatten(),
            min(max(self.config.intrinsic_rescue_texture_quantile, 0.0), 1.0),
        ).detach().clamp_min(1e-5)
        texture_evidence = torch.sigmoid(
            (intrinsic_texture - texture_cut)
            / texture_cut
            * self.config.water_null_texture_sharpness
        )

        intrinsic_ambient_distance = torch.abs(
            intrinsic - water_background_image[..., :3].detach()
        ).mean(dim=-1, keepdim=True)
        color_residual_evidence = 1.0 - torch.exp(
            -intrinsic_ambient_distance / max(self.config.intrinsic_rescue_color_sigma, 1e-4)
        )

        # A simple non-blue / non-free-water chroma cue. It is only allowed to
        # rescue visible intrinsic content, so near-black pure water does not
        # become scene evidence just because it differs from the blue ambient.
        red_or_green = torch.maximum(intrinsic[..., 0:1], intrinsic[..., 1:2])
        blue = intrinsic[..., 2:3]
        non_blue_evidence = torch.sigmoid((red_or_green - 0.85 * blue) / 0.05)
        appearance_evidence = luma_evidence * torch.maximum(color_residual_evidence, non_blue_evidence)
        blue_water_chroma = self._intrinsic_blue_water_chroma(intrinsic)
        appearance_evidence = appearance_evidence * (
            1.0 - self.config.intrinsic_rescue_blue_suppression * blue_water_chroma
        ).clamp(0.0, 1.0)

        if depth_edge_support is None:
            depth_edge_support = torch.zeros_like(reference)
        geometry_evidence = depth_edge_support.detach().clamp(0.0, 1.0)

        structure_evidence = torch.maximum(texture_evidence, geometry_evidence)
        structure_strength = max(min(self.config.intrinsic_rescue_structure_strength, 1.0), 0.0)
        rescue = appearance_evidence * (1.0 + structure_strength * structure_evidence)

        if accumulation is not None:
            alpha = torch.nan_to_num(
                accumulation[..., :1].detach(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            alpha_threshold = max(self.config.intrinsic_rescue_alpha_threshold, 1e-4)
            alpha_evidence = torch.sigmoid((alpha - alpha_threshold) / alpha_threshold * 6.0)
            rescue = rescue * (0.35 + 0.65 * alpha_evidence)

        # Expand only appearance-backed scene evidence by one pixel. This keeps
        # fragile object boundaries, while blue water streaks / depth-edge
        # artifacts cannot be rescued by structure alone.
        rescue_chw = rescue.permute(2, 0, 1).unsqueeze(0)
        rescue_dilated = F.max_pool2d(rescue_chw, kernel_size=3, stride=1, padding=1)
        rescue = torch.maximum(rescue, 0.45 * rescue_dilated.squeeze(0).permute(1, 2, 0))
        return rescue.detach().clamp(0.0, 1.0)

    def _build_horizon_prior(
        self,
        base_transition: torch.Tensor,
        ambient_confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate a per-image free-water boundary prior from top-connected transition evidence."""
        if not self.config.enable_water_horizon_prior:
            return torch.zeros_like(base_transition)
        if base_transition.numel() == 0 or base_transition.dim() < 3:
            return torch.zeros_like(base_transition)

        score = (base_transition * ambient_confidence).detach().squeeze(-1).clamp(0.0, 1.0)
        height, width = score.shape[:2]
        top_rows = max(1, int(height * 0.20))
        top_strength = score[:top_rows, :].mean(dim=0)
        if float(top_strength.mean().detach().cpu()) < self.config.water_horizon_min_confidence:
            return torch.zeros_like(base_transition)

        y = torch.linspace(0.0, 1.0, height, device=score.device, dtype=score.dtype)
        top_bias = (1.0 - y).clamp_min(0.0)[:, None]
        weights = score * top_bias
        boundary = (weights * y[:, None]).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-6)
        boundary = (boundary + self.config.water_horizon_band).clamp(0.0, 1.0)

        band = max(self.config.water_horizon_band, 1e-3)
        prior = torch.sigmoid((boundary[None, :] - y[:, None]) / band * 6.0)
        prior = prior * top_strength[None, :].clamp(0.0, 1.0) * self.config.water_horizon_strength
        return prior[..., None].clamp(0.0, 1.0)

    def _build_water_null_probabilities(
        self,
        pseudo_depth: torch.Tensor,
        gt_underwater_image: torch.Tensor,
        water_background_image: torch.Tensor,
        intrinsic_image: Optional[torch.Tensor] = None,
        accumulation: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Estimate AquaNull-v2 pure, transition, and combined null-space probabilities.

        DepthAnything inputs in this project are normalized disparity-like maps: near
        content is large, distant / water-only content is small. Pure water is a
        strict cue intersection; transition water additionally admits blue-gray,
        low-edge free-water boundaries through a horizon prior.
        """
        depth = pseudo_depth[..., :1].detach()
        depth = torch.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        valid_depth = depth[torch.isfinite(depth)]
        if valid_depth.numel() == 0:
            zero = torch.zeros_like(depth)
            return {
                "pure": zero,
                "transition": zero,
                "null": zero,
                "horizon": zero,
                "depth_edge_support": zero,
                "color_residual_support": zero,
                "scene_rescue": zero,
                "residual_water": zero,
                "connected_freewater": zero,
            }

        static_cut = torch.tensor(
            self.config.water_null_depth_threshold,
            device=depth.device,
            dtype=depth.dtype,
        )
        adaptive_cut = torch.quantile(
            valid_depth.flatten(),
            min(max(self.config.water_null_depth_quantile, 0.0), 1.0),
        ).detach()
        depth_cut = torch.maximum(static_cut, adaptive_cut).clamp_min(1e-4)
        depth_confidence = torch.sigmoid(
            (depth_cut - depth)
            / depth_cut
            * self.config.water_null_depth_sharpness
        )
        transition_cut = torch.quantile(
            valid_depth.flatten(),
            min(max(self.config.water_transition_depth_quantile, 0.0), 1.0),
        ).detach().clamp_min(depth_cut + 1e-4)
        transition_depth_confidence = torch.sigmoid(
            (transition_cut - depth)
            / transition_cut
            * self.config.water_null_depth_sharpness
        )

        texture = self._image_gradient_magnitude(gt_underwater_image).detach()
        texture_cut = torch.quantile(
            texture.flatten(),
            min(max(self.config.water_null_texture_quantile, 0.0), 1.0),
        ).detach().clamp_min(1e-5)
        texture_confidence = torch.sigmoid(
            (texture_cut - texture)
            / texture_cut
            * self.config.water_null_texture_sharpness
        )
        if intrinsic_image is not None:
            intrinsic_texture = self._image_gradient_magnitude(intrinsic_image.detach()).detach()
            intrinsic_texture_cut = torch.quantile(
                intrinsic_texture.flatten(),
                min(max(self.config.water_transition_intrinsic_texture_quantile, 0.0), 1.0),
            ).detach().clamp_min(1e-5)
            intrinsic_texture_confidence = torch.sigmoid(
                (intrinsic_texture_cut - intrinsic_texture)
                / intrinsic_texture_cut
                * self.config.water_null_texture_sharpness
            )
        else:
            intrinsic_texture = None
            intrinsic_texture_confidence = torch.ones_like(texture_confidence)

        ambient_distance = torch.abs(
            gt_underwater_image[..., :3].detach() - water_background_image.detach()
        ).mean(dim=-1, keepdim=True)
        ambient_confidence = torch.exp(
            -ambient_distance / max(self.config.water_null_ambient_color_sigma, 1e-4)
        )
        transition_ambient_confidence = torch.exp(
            -ambient_distance / max(self.config.water_transition_ambient_color_sigma, 1e-4)
        )

        edge_cut = torch.quantile(
            texture.flatten(),
            min(max(self.config.water_null_edge_quantile, 0.0), 1.0),
        ).detach().clamp_min(1e-5)
        edge_confidence = torch.sigmoid(
            (texture - edge_cut)
            / edge_cut
            * self.config.water_null_edge_sharpness
        )
        edge_protection = 1.0 - max(min(self.config.water_null_edge_protection, 1.0), 0.0) * edge_confidence

        depth_edge = self._image_gradient_magnitude(depth).detach()
        depth_edge_cut = torch.quantile(
            depth_edge.flatten(),
            min(max(self.config.water_depth_edge_quantile, 0.0), 1.0),
        ).detach().clamp_min(1e-5)
        low_depth_edge_confidence = torch.sigmoid(
            (depth_edge_cut - depth_edge)
            / depth_edge_cut
            * self.config.water_depth_edge_sharpness
        )
        depth_edge_support = torch.sigmoid(
            (depth_edge - depth_edge_cut)
            / depth_edge_cut
            * self.config.water_depth_edge_sharpness
        ).detach().clamp(0.0, 1.0)
        color_residual_support = (ambient_distance / max(self.config.water_transition_ambient_color_sigma, 1e-4)).detach().clamp(0.0, 1.0)
        scene_rescue = self._build_intrinsic_scene_rescue(
            intrinsic_image,
            water_background_image,
            intrinsic_texture,
            depth_edge_support,
            accumulation,
        )

        pure_base = depth_confidence * texture_confidence
        pure_base = pure_base * (ambient_confidence * ambient_confidence)
        pure_base = pure_base * edge_protection
        pure_probability = pure_base
        pure_probability = pure_probability * (
            1.0 - self.config.intrinsic_rescue_pure_strength * scene_rescue
        ).clamp(0.0, 1.0)

        transition_base = transition_depth_confidence * texture_confidence
        transition_base = transition_base * intrinsic_texture_confidence
        transition_base = transition_base * transition_ambient_confidence
        transition_base = transition_base * low_depth_edge_confidence
        transition_base = transition_base * edge_protection.clamp_min(0.15)
        horizon_prior = self._build_horizon_prior(transition_base, transition_ambient_confidence)
        transition_probability = torch.maximum(
            transition_base,
            horizon_prior * transition_depth_confidence * low_depth_edge_confidence,
        )
        transition_probability = transition_probability * (
            1.0 - self.config.intrinsic_rescue_transition_strength * scene_rescue
        ).clamp(0.0, 1.0)

        if intrinsic_image is not None:
            intrinsic_blue_water = self._intrinsic_blue_water_chroma(intrinsic_image)
            intrinsic = torch.nan_to_num(
                intrinsic_image[..., :3].detach(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            intrinsic_ambient_distance = torch.abs(
                intrinsic - water_background_image[..., :3].detach()
            ).mean(dim=-1, keepdim=True)
            intrinsic_water_similarity = torch.exp(
                -intrinsic_ambient_distance / max(self.config.intrinsic_rescue_color_sigma, 1e-4)
            )
        else:
            intrinsic_blue_water = torch.zeros_like(depth)
            intrinsic_water_similarity = torch.zeros_like(depth)

        # Residual water is blue/cyan haze that survives in raw intrinsic. It is
        # not the same as pure water: suppress water color first, without
        # treating all structure as true scene.
        residual_scene_guard_floor = max(
            min(self.config.residual_water_scene_guard_floor, 1.0),
            0.0,
        )
        residual_scene_guard = (
            1.0 - 0.50 * self.config.residual_water_scene_suppression * scene_rescue
        ).clamp(residual_scene_guard_floor, 1.0)
        residual_depth_floor = max(min(self.config.residual_water_depth_floor, 1.0), 0.0)
        residual_similarity_floor = max(min(self.config.residual_water_similarity_floor, 1.0), 0.0)
        residual_depth_context = residual_depth_floor + (
            1.0 - residual_depth_floor
        ) * transition_depth_confidence
        residual_similarity_context = residual_similarity_floor + (
            1.0 - residual_similarity_floor
        ) * intrinsic_water_similarity
        residual_water_probability = intrinsic_blue_water
        residual_water_probability = residual_water_probability * residual_depth_context
        residual_water_probability = residual_water_probability * residual_similarity_context
        residual_water_probability = residual_water_probability * residual_scene_guard
        residual_water_probability = torch.maximum(
            residual_water_probability,
            intrinsic_blue_water * torch.maximum(pure_base, transition_probability).detach(),
        )
        residual_water_probability = residual_water_probability.detach().clamp(0.0, 1.0)

        connected_seed = torch.maximum(pure_base.detach(), residual_water_probability)
        connected_support = torch.maximum(transition_base.detach(), residual_water_probability)
        connected_support = connected_support * (
            1.0 - 0.50 * scene_rescue
        ).clamp(0.15, 1.0)
        connected_freewater = self._build_connected_freewater_prior(
            connected_seed.clamp(0.0, 1.0),
            connected_support.clamp(0.0, 1.0),
        )

        null_probability = torch.maximum(pure_probability, transition_probability)
        null_probability = torch.maximum(
            null_probability,
            residual_water_probability * self.config.residual_water_strength,
        )
        null_probability = torch.maximum(
            null_probability,
            connected_freewater * self.config.connected_freewater_strength,
        )

        return {
            "pure": pure_probability.detach().clamp(0.0, 1.0),
            "transition": transition_probability.detach().clamp(0.0, 1.0),
            "null": null_probability.detach().clamp(0.0, 1.0),
            "horizon": horizon_prior.detach().clamp(0.0, 1.0),
            "depth_edge_support": depth_edge_support,
            "color_residual_support": color_residual_support,
            "scene_rescue": scene_rescue,
            "residual_water": residual_water_probability,
            "connected_freewater": connected_freewater.detach().clamp(0.0, 1.0),
        }

    def _build_water_null_probability(
        self,
        pseudo_depth: torch.Tensor,
        gt_underwater_image: torch.Tensor,
        water_background_image: torch.Tensor,
        intrinsic_image: Optional[torch.Tensor] = None,
        accumulation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compatibility wrapper returning the combined AquaNull-v2 null probability."""
        return self._build_water_null_probabilities(
            pseudo_depth,
            gt_underwater_image,
            water_background_image,
            intrinsic_image,
            accumulation,
        )["null"]

    def _build_strict_water_null_probability(
        self,
        water_probability: torch.Tensor,
        threshold: Optional[float] = None,
        max_fraction: Optional[float] = None,
    ) -> torch.Tensor:
        """Return high-confidence pure-water weights for destructive null losses."""
        threshold = self.config.water_null_strict_threshold if threshold is None else threshold
        threshold = max(min(threshold, 0.999), 0.0)
        strict_probability = torch.where(
            water_probability >= threshold,
            water_probability,
            torch.zeros_like(water_probability),
        )

        max_fraction = self.config.water_pure_max_fraction if max_fraction is None else max_fraction
        if 0.0 < max_fraction < 1.0 and strict_probability.numel() > 0:
            cutoff = torch.quantile(
                water_probability.flatten(),
                1.0 - max(min(max_fraction, 1.0), 0.0),
            ).detach()
            cutoff = torch.maximum(
                cutoff,
                torch.tensor(threshold, device=water_probability.device, dtype=water_probability.dtype),
            )
            strict_probability = torch.where(
                water_probability >= cutoff,
                strict_probability,
                torch.zeros_like(strict_probability),
            )

        return strict_probability.detach().clamp(0.0, 1.0)

    def _build_depth_foreground_mask(
        self,
        pseudo_depth: torch.Tensor,
        image_idx,
    ) -> Tuple[torch.Tensor, float]:
        """Original SeaFree-GS foreground mask from pseudo-depth contours."""
        downscale = self._get_downscale_factor()
        if image_idx in self.foreground_mask_cache.keys() and downscale in self.foreground_mask_cache[image_idx].keys():
            cached = self.foreground_mask_cache[image_idx][downscale]
            return cached["mask_tensor"], cached["background_pixel_ratio"]

        background_depth_threshold = 1e-2
        pseudo_depth_numpy = pseudo_depth.squeeze().detach().cpu().numpy()
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
        self.foreground_mask_cache[image_idx][downscale] = {
            "mask_tensor": foreground_mask,
            "background_pixel_ratio": background_pixel_ratio,
        }
        return foreground_mask, background_pixel_ratio

    def _ensure_water_lifecycle_buffers(self):
        """Keep per-Gaussian water/scene exposure buffers aligned with current params."""
        n = self.num_points
        device = self.means.device
        if not hasattr(self, "strategy_state"):
            return
        if (
            "water_exposure_score" not in self.strategy_state
            or self.strategy_state["water_exposure_score"].shape[0] != n
            or self.strategy_state["water_exposure_score"].device != device
        ):
            self.strategy_state["water_exposure_score"] = torch.zeros(n, device=device)
        if (
            "scene_exposure_score" not in self.strategy_state
            or self.strategy_state["scene_exposure_score"].shape[0] != n
            or self.strategy_state["scene_exposure_score"].device != device
        ):
            self.strategy_state["scene_exposure_score"] = torch.zeros(n, device=device)
        if (
            "depth_edge_support_score" not in self.strategy_state
            or self.strategy_state["depth_edge_support_score"].shape[0] != n
            or self.strategy_state["depth_edge_support_score"].device != device
        ):
            self.strategy_state["depth_edge_support_score"] = torch.zeros(n, device=device)
        if (
            "color_residual_support_score" not in self.strategy_state
            or self.strategy_state["color_residual_support_score"].shape[0] != n
            or self.strategy_state["color_residual_support_score"].device != device
        ):
            self.strategy_state["color_residual_support_score"] = torch.zeros(n, device=device)
        if (
            "view_consistency_score" not in self.strategy_state
            or self.strategy_state["view_consistency_score"].shape[0] != n
            or self.strategy_state["view_consistency_score"].device != device
        ):
            self.strategy_state["view_consistency_score"] = torch.zeros(n, device=device)
        self.water_exposure_score = self.strategy_state["water_exposure_score"]
        self.scene_exposure_score = self.strategy_state["scene_exposure_score"]
        self.depth_edge_support_score = self.strategy_state["depth_edge_support_score"]
        self.color_residual_support_score = self.strategy_state["color_residual_support_score"]
        self.view_consistency_score = self.strategy_state["view_consistency_score"]

    @torch.no_grad()
    def _update_water_lifecycle_scores(
        self,
        water_probability: torch.Tensor,
        depth_edge_support: Optional[torch.Tensor] = None,
        color_residual_support: Optional[torch.Tensor] = None,
    ):
        """Accumulate per-Gaussian evidence of whether a Gaussian explains water-only rays."""
        if not self.training or not self.config.enable_water_null_occupancy:
            return
        if not hasattr(self, "info") or self.info is None:
            return
        means2d = self.info.get("means2d")
        radii = self.info.get("radii")
        opacities = self.info.get("opacities")
        if means2d is None or radii is None or means2d.dim() != 3:
            return

        self._ensure_water_lifecycle_buffers()
        water_score = self.strategy_state["water_exposure_score"]
        scene_score = self.strategy_state["scene_exposure_score"]
        depth_edge_score = self.strategy_state["depth_edge_support_score"]
        color_residual_score = self.strategy_state["color_residual_support_score"]
        view_consistency_score = self.strategy_state["view_consistency_score"]
        height, width = water_probability.shape[:2]
        if depth_edge_support is None:
            depth_edge_support = torch.zeros_like(water_probability)
        if color_residual_support is None:
            color_residual_support = torch.zeros_like(water_probability)
        xy = means2d[0].detach()
        radii0 = radii[0].detach()
        x_float = xy[:, 0]
        y_float = xy[:, 1]
        in_bounds = (
            (radii0 > 0)
            & torch.isfinite(x_float)
            & torch.isfinite(y_float)
            & (x_float >= 0)
            & (x_float < width)
            & (y_float >= 0)
            & (y_float < height)
        )
        if not torch.any(in_bounds):
            return

        # Approximate each projected splat footprint with five water-probability
        # samples. This keeps the module Python-side while avoiding a brittle
        # center-only attribution for large / elongated water splats.
        water_sum = torch.zeros_like(x_float, device=water_score.device)
        edge_sum = torch.zeros_like(x_float, device=water_score.device)
        residual_sum = torch.zeros_like(x_float, device=water_score.device)
        sample_count = torch.zeros_like(x_float, device=water_score.device)
        radius_offset = radii0.to(dtype=x_float.dtype).clamp_min(1.0) * 0.5
        for offset_x, offset_y in (
            (0.0, 0.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
        ):
            sample_x = x_float + radius_offset * offset_x
            sample_y = y_float + radius_offset * offset_y
            sample_valid = (
                in_bounds
                & (sample_x >= 0)
                & (sample_x < width)
                & (sample_y >= 0)
                & (sample_y < height)
            )
            x = sample_x.round().long().clamp(0, width - 1)
            y = sample_y.round().long().clamp(0, height - 1)
            sampled_water = water_probability[y, x, 0].to(water_score.device)
            sampled_edge = depth_edge_support[y, x, 0].to(water_score.device)
            sampled_residual = color_residual_support[y, x, 0].to(water_score.device)
            sample_valid_float = sample_valid.to(water_score.device)
            water_sum += sampled_water * sample_valid_float
            edge_sum += sampled_edge * sample_valid_float
            residual_sum += sampled_residual * sample_valid_float
            sample_count += sample_valid_float
        water_at_mean = water_sum / sample_count.clamp_min(1.0)
        edge_at_mean = edge_sum / sample_count.clamp_min(1.0)
        residual_at_mean = residual_sum / sample_count.clamp_min(1.0)
        if opacities is not None and opacities.dim() >= 2:
            visibility_weight = opacities[0].detach().flatten().to(water_at_mean.device)
        else:
            visibility_weight = torch.sigmoid(self.opacities.detach().flatten()).to(water_at_mean.device)
        visibility_weight = torch.where(in_bounds.to(water_at_mean.device), visibility_weight, torch.zeros_like(visibility_weight))

        ema = min(max(self.config.water_lifecycle_ema, 0.0), 0.999)
        update = 1.0 - ema
        observed = in_bounds.to(water_score.device)
        water_update = (water_at_mean * visibility_weight).clamp(0.0, 1.0)
        scene_update = ((1.0 - water_at_mean) * visibility_weight).clamp(0.0, 1.0)
        edge_update = (edge_at_mean * visibility_weight).clamp(0.0, 1.0)
        residual_update = (residual_at_mean * visibility_weight).clamp(0.0, 1.0)
        consistency_update = ((water_at_mean - 0.5).abs() * 2.0 * visibility_weight).clamp(0.0, 1.0)
        water_score = torch.where(
            observed,
            ema * water_score + update * water_update,
            water_score,
        )
        scene_score = torch.where(
            observed,
            ema * scene_score + update * scene_update,
            scene_score,
        )
        depth_edge_score = torch.where(
            observed,
            ema * depth_edge_score + update * edge_update,
            depth_edge_score,
        )
        color_residual_score = torch.where(
            observed,
            ema * color_residual_score + update * residual_update,
            color_residual_score,
        )
        view_consistency_score = torch.where(
            observed,
            ema * view_consistency_score + update * consistency_update,
            view_consistency_score,
        )
        self.strategy_state["water_exposure_score"] = water_score
        self.strategy_state["scene_exposure_score"] = scene_score
        self.strategy_state["depth_edge_support_score"] = depth_edge_score
        self.strategy_state["color_residual_support_score"] = color_residual_score
        self.strategy_state["view_consistency_score"] = view_consistency_score
        self.water_exposure_score = water_score
        self.scene_exposure_score = scene_score
        self.depth_edge_support_score = depth_edge_score
        self.color_residual_support_score = color_residual_score
        self.view_consistency_score = view_consistency_score

    def _water_dominated_mask(self, score_threshold: float, ratio: float) -> Optional[torch.Tensor]:
        if not self.config.enable_water_null_occupancy:
            return None
        self._ensure_water_lifecycle_buffers()
        water_score = self.strategy_state["water_exposure_score"].detach()
        scene_score = self.strategy_state["scene_exposure_score"].detach()
        depth_edge_score = self.strategy_state["depth_edge_support_score"].detach()
        color_residual_score = self.strategy_state["color_residual_support_score"].detach()
        view_consistency_score = self.strategy_state["view_consistency_score"].detach()
        return (
            (water_score > score_threshold)
            & (water_score > scene_score * ratio)
            & (depth_edge_score < self.config.water_depth_edge_support_max)
            & (color_residual_score < self.config.water_color_residual_support_max)
            & (view_consistency_score > self.config.water_view_consistency_min)
        )

    def _get_water_lifecycle_no_grow_mask(self) -> Optional[torch.Tensor]:
        """Return Gaussians that should be hard-blocked from duplicate / split."""
        if (
            not self.training
            or not self.config.enable_water_null_occupancy
            or not self.config.enable_water_lifecycle_growth_suppression
            or self.step < self.config.water_growth_suppress_start_step
            or self.step % self.config.refine_every != 0
        ):
            return None
        return self._water_dominated_mask(
            self.config.water_growth_suppress_threshold,
            self.config.water_growth_suppress_ratio,
        )

    def _suppress_water_dominated_growth(self):
        """Reduce 2D densification gradients for Gaussians mostly explaining pure water."""
        if (
            not self.training
            or not self.config.enable_water_null_occupancy
            or not self.config.enable_water_lifecycle_growth_suppression
            or self.step < self.config.water_growth_suppress_start_step
            or not hasattr(self, "info")
            or self.info is None
        ):
            return
        water_dominated = self._water_dominated_mask(
            self.config.water_growth_suppress_threshold,
            self.config.water_growth_suppress_ratio,
        )
        if water_dominated is None or not torch.any(water_dominated):
            return
        means2d = self.info.get("means2d")
        if means2d is None or means2d.dim() != 3 or means2d.shape[1] != water_dominated.shape[0]:
            return
        scale = self.config.water_growth_grad_scale
        if means2d.grad is not None:
            means2d.grad[:, water_dominated, :] *= scale
        if hasattr(means2d, "absgrad") and means2d.absgrad is not None:
            means2d.absgrad[:, water_dominated, :] *= scale

    def _get_water_lifecycle_prune_mask(self) -> Optional[torch.Tensor]:
        """Return a capped prune mask for Gaussians that consistently explain water null-space."""
        if (
            not self.training
            or not self.config.enable_water_null_occupancy
            or not self.config.enable_water_lifecycle_pruning
            or self.step < self.config.water_prune_start_step
            or self.step % self.config.refine_every != 0
        ):
            return None
        self._ensure_water_lifecycle_buffers()
        prune_mask = self._water_dominated_mask(
            self.config.water_prune_score_threshold,
            self.config.water_prune_score_ratio,
        )
        if prune_mask is None:
            return None
        water_score = self.strategy_state["water_exposure_score"].detach()
        scene_score = self.strategy_state["scene_exposure_score"].detach()
        prune_mask = prune_mask & (scene_score < self.config.water_prune_scene_score_max)
        n_prune = int(prune_mask.sum().item())
        if n_prune == 0:
            return prune_mask

        max_prune = max(1, int(self.num_points * max(self.config.water_prune_max_fraction, 0.0)))
        if n_prune > max_prune:
            ranking = water_score - scene_score
            candidate_ids = torch.where(prune_mask)[0]
            keep_ids = candidate_ids[torch.topk(ranking[candidate_ids], k=max_prune, largest=True).indices]
            capped_mask = torch.zeros_like(prune_mask)
            capped_mask[keep_ids] = True
            prune_mask = capped_mask
        return prune_mask

    def _sync_water_lifecycle_buffers(self, n_before_refine: int):
        """Mirror lifecycle scores after gsplat changes the Gaussian table length."""
        self._ensure_water_lifecycle_buffers()

    def _gaussian_null_dominance_score(self) -> torch.Tensor:
        """Return per-Gaussian null dominance used for rendering gates and hard blocks."""
        self._ensure_water_lifecycle_buffers()
        water_score = self.strategy_state["water_exposure_score"].detach()
        scene_score = self.strategy_state["scene_exposure_score"].detach()
        edge_score = self.strategy_state["depth_edge_support_score"].detach()
        residual_score = self.strategy_state["color_residual_support_score"].detach()
        consistency_score = self.strategy_state["view_consistency_score"].detach()
        null_ratio = water_score / (water_score + scene_score + 1e-6)
        edge_guard = (1.0 - edge_score / max(self.config.water_depth_edge_support_max, 1e-4)).clamp(0.0, 1.0)
        residual_guard = (1.0 - residual_score / max(self.config.water_color_residual_support_max, 1e-4)).clamp(0.0, 1.0)
        consistency_guard = (0.25 + 0.75 * consistency_score).clamp(0.0, 1.0)
        return (null_ratio * edge_guard * residual_guard * consistency_guard).clamp(0.0, 1.0).unsqueeze(-1)

    def _build_intrinsic_visibility_gate(
        self,
        alpha: torch.Tensor,
        water_probability: Optional[torch.Tensor] = None,
        alpha_water: Optional[torch.Tensor] = None,
        scene_rescue: Optional[torch.Tensor] = None,
        residual_water: Optional[torch.Tensor] = None,
        connected_freewater: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scene visibility gate shared by intrinsic rendering and null-space losses."""
        if water_probability is None:
            water_probability = torch.zeros_like(alpha)
        if alpha_water is None:
            alpha_water = torch.zeros_like(alpha)
        if scene_rescue is None:
            scene_rescue = torch.zeros_like(alpha)
        if residual_water is None:
            residual_water = torch.zeros_like(alpha)
        if connected_freewater is None:
            connected_freewater = torch.zeros_like(alpha)

        water_probability = torch.nan_to_num(
            water_probability.detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        alpha = torch.nan_to_num(alpha.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        alpha_water = torch.nan_to_num(
            alpha_water.detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        scene_rescue = torch.nan_to_num(
            scene_rescue.detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        residual_water = torch.nan_to_num(
            residual_water.detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        connected_freewater = torch.nan_to_num(
            connected_freewater.detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        water_alpha_ratio = (alpha_water / alpha.clamp_min(1e-6)).clamp(0.0, 1.0)
        water_strength = torch.maximum(water_probability, water_alpha_ratio)
        residual_visibility = 1.0 - torch.pow(
            (1.0 - residual_water).clamp(0.0, 1.0),
            max(self.config.residual_water_visibility_gamma, 1e-4),
        )
        connected_visibility = 1.0 - torch.pow(
            (1.0 - connected_freewater).clamp(0.0, 1.0),
            max(self.config.connected_freewater_visibility_gamma, 1e-4),
        )
        water_strength = torch.maximum(
            water_strength,
            residual_visibility * max(min(self.config.residual_water_visibility_strength, 1.0), 0.0),
        )
        water_strength = torch.maximum(
            water_strength,
            connected_visibility * max(min(self.config.connected_freewater_visibility_strength, 1.0), 0.0),
        )
        freewater_rescue_cancel = torch.maximum(
            residual_water,
            connected_freewater * 0.75,
        )
        scene_rescue_safe = scene_rescue * (
            1.0 - self.config.residual_water_scene_suppression * freewater_rescue_cancel
        ).clamp(0.0, 1.0)
        rescue_strength = max(min(self.config.intrinsic_visibility_rescue_strength, 1.0), 0.0)
        visibility = 1.0 - water_strength * (1.0 - rescue_strength * scene_rescue_safe)
        visibility_floor = max(min(self.config.intrinsic_visibility_floor, 1.0), 0.0) * scene_rescue_safe
        visibility = torch.maximum(visibility, visibility_floor)
        return visibility.clamp(0.0, 1.0)

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
            "intrinsic_color_render_raw": rgb.clone(),
            "intrinsic_visibility": torch.ones_like(accumulation),
            "water_accumulation": accumulation.clone(),
            "rendered_null_probability": accumulation.clone(),
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

        if self.config.enable_water_null_occupancy:
            gaussian_null_scores = self._gaussian_null_dominance_score()
        else:
            gaussian_null_scores = torch.zeros_like(self.opacities)
        gaussian_scene_scores = (1.0 - gaussian_null_scores).clamp(0.0, 1.0)
        if crop_ids is not None:
            gaussian_null_scores_crop = gaussian_null_scores[crop_ids]
            gaussian_scene_scores_crop = gaussian_scene_scores[crop_ids]
        else:
            gaussian_null_scores_crop = gaussian_null_scores
            gaussian_scene_scores_crop = gaussian_scene_scores

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
        colors_crop = torch.cat(
            (
                degraded_gaussian_colors,
                colors_crop,
                gaussian_null_scores_crop.to(colors_crop),
                gaussian_scene_scores_crop.to(colors_crop),
            ),
            dim=-1,
        )

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

        intrinsic_color_render_raw = torch.clamp(render[:, ..., 3:6], 0.0, 1.0)
        water_accumulation = render[:, ..., 6:7].clamp(0.0, 1.0)
        rendered_null_probability = (water_accumulation / alpha.clamp_min(1e-6)).clamp(0.0, 1.0)
        intrinsic_visibility = self._build_intrinsic_visibility_gate(
            alpha,
            rendered_null_probability,
            water_accumulation,
        )
        intrinsic_color_render_clamped = intrinsic_color_render_raw * intrinsic_visibility


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
            "intrinsic_color_render_raw": intrinsic_color_render_raw.squeeze(0),  # type: ignore
            "intrinsic_visibility": intrinsic_visibility.squeeze(0),  # type: ignore
            "water_accumulation": water_accumulation.squeeze(0),  # type: ignore
            "rendered_null_probability": rendered_null_probability.squeeze(0),  # type: ignore
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
        if self.config.enable_water_null_occupancy:
            self._ensure_water_lifecycle_buffers()
            metrics_dict["water_null_probability_mean"] = self.last_water_probability_mean.to(self.device)
            metrics_dict["water_pure_probability_mean"] = self.last_water_pure_probability_mean.to(self.device)
            metrics_dict["water_transition_probability_mean"] = self.last_water_transition_probability_mean.to(self.device)
            metrics_dict["water_null_strict_probability_mean"] = self.last_water_strict_probability_mean.to(self.device)
            metrics_dict["water_horizon_probability_mean"] = self.last_water_horizon_probability_mean.to(self.device)
            metrics_dict["residual_water_probability_mean"] = self.last_residual_water_probability_mean.to(self.device)
            metrics_dict["connected_freewater_probability_mean"] = self.last_connected_freewater_probability_mean.to(self.device)
            metrics_dict["water_exposure_score_mean"] = self.strategy_state["water_exposure_score"].mean()
            metrics_dict["scene_exposure_score_mean"] = self.strategy_state["scene_exposure_score"].mean()
            metrics_dict["depth_edge_support_score_mean"] = self.strategy_state["depth_edge_support_score"].mean()
            metrics_dict["color_residual_support_score_mean"] = self.strategy_state["color_residual_support_score"].mean()
            metrics_dict["view_consistency_score_mean"] = self.strategy_state["view_consistency_score"].mean()
            metrics_dict["water_no_grow_candidates"] = torch.tensor(
                float(self.last_water_no_grow_candidates),
                device=self.device,
            )
            metrics_dict["water_prune_candidates"] = torch.tensor(
                float(self.last_water_prune_candidates),
                device=self.device,
            )

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
            
        water_null_probability = None
        water_pure_probability = None
        water_transition_probability = None
        strict_water_null_probability = None
        strict_pure_water_probability = None
        strict_transition_water_probability = None
        strict_destructive_water_probability = None
        foreground_mask, background_pixel_ratio = self._build_depth_foreground_mask(
            pseudo_depth,
            batch["image_idx"],
        )
        if self.config.enable_water_null_occupancy:
            water_probabilities = self._build_water_null_probabilities(
                pseudo_depth,
                gt_underwater_image,
                outputs["water_background_image"],
                outputs.get("intrinsic_color_render_raw", outputs["intrinsic_color_render"]),
                outputs["accumulation"],
            )
            water_pure_probability = water_probabilities["pure"]
            water_transition_probability = water_probabilities["transition"]
            water_null_probability = water_probabilities["null"]
            strict_pure_water_probability = self._build_strict_water_null_probability(
                water_pure_probability,
                self.config.water_null_strict_threshold,
                self.config.water_pure_max_fraction,
            )
            strict_transition_water_probability = self._build_strict_water_null_probability(
                water_transition_probability,
                self.config.water_transition_strict_threshold,
                self.config.water_transition_max_fraction,
            )
            strict_water_null_probability = torch.maximum(strict_pure_water_probability, strict_transition_water_probability)
            strict_destructive_water_probability = strict_pure_water_probability
            self.last_water_probability_mean = water_null_probability.mean().detach()
            self.last_water_pure_probability_mean = water_pure_probability.mean().detach()
            self.last_water_transition_probability_mean = water_transition_probability.mean().detach()
            self.last_water_strict_probability_mean = strict_water_null_probability.mean().detach()
            self.last_water_horizon_probability_mean = water_probabilities["horizon"].mean().detach()
            self.last_residual_water_probability_mean = water_probabilities["residual_water"].mean().detach()
            self.last_connected_freewater_probability_mean = water_probabilities["connected_freewater"].mean().detach()
            self._update_water_lifecycle_scores(
                strict_destructive_water_probability,
                water_probabilities["depth_edge_support"],
                water_probabilities["color_residual_support"],
            )
        
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
            used_strict_water_background = False
            if self.config.enable_water_null_occupancy and strict_destructive_water_probability is not None:
                if float((strict_destructive_water_probability > 0).float().mean().detach().cpu()) > 0.005:
                    background_ambient_light_weight = 1 / (water_background_image.detach() + 1e-3)
                    background_water_supervision_loss += self._weighted_mean(
                        torch.abs((water_background_image - gt_underwater_image) * background_ambient_light_weight),
                        strict_destructive_water_probability.repeat(1, 1, 3),
                    )
                    used_strict_water_background = True
            if not used_strict_water_background and background_pixel_ratio > 0.05:
                background_ambient_light_pixels = water_background_image[foreground_mask.repeat(1, 1, 3) < 0.5]
                background_underwater_pixels = gt_underwater_image[foreground_mask.repeat(1, 1, 3) < 0.5]            
                background_ambient_light_weight = 1 / (background_ambient_light_pixels.detach() + 1e-3)
                background_water_supervision_loss += torch.abs(
                    (background_ambient_light_pixels - background_underwater_pixels) * background_ambient_light_weight
                ).mean()        

        water_empty_occupancy_loss = torch.tensor(0.0, device=self.device)
        water_transition_alpha_loss = torch.tensor(0.0, device=self.device)
        water_intrinsic_null_loss = torch.tensor(0.0, device=self.device)
        water_background_ownership_loss = torch.tensor(0.0, device=self.device)
        residual_water_chroma_loss = torch.tensor(0.0, device=self.device)
        if (
            self.config.enable_water_null_occupancy
            and strict_destructive_water_probability is not None
            and self.step >= self.config.water_null_start_step
        ):
            water_empty_occupancy_loss = self._weighted_mean(
                outputs["accumulation"],
                strict_pure_water_probability,
            )
            water_transition_alpha_loss = self._weighted_mean(
                torch.relu(outputs["accumulation"] - self.config.water_transition_alpha_target),
                strict_transition_water_probability,
            )
            water_intrinsic_null_loss = self._weighted_mean(
                torch.abs(outputs.get("intrinsic_color_render_raw", outputs["intrinsic_color_render"])),
                strict_destructive_water_probability.repeat(1, 1, 3),
            )
            water_background_ownership_loss = self._weighted_mean(
                torch.abs(rendered_underwater_image - water_background_image),
                strict_destructive_water_probability.repeat(1, 1, 3),
            )
            intrinsic_raw_for_chroma = outputs.get("intrinsic_color_render_raw", outputs["intrinsic_color_render"])
            blue_residual = torch.relu(
                intrinsic_raw_for_chroma[..., 2:3]
                - torch.maximum(intrinsic_raw_for_chroma[..., 0:1], intrinsic_raw_for_chroma[..., 1:2])
                - self.config.intrinsic_water_blue_margin
            )
            cyan_residual = torch.relu(
                torch.minimum(intrinsic_raw_for_chroma[..., 1:2], intrinsic_raw_for_chroma[..., 2:3])
                - intrinsic_raw_for_chroma[..., 0:1]
                - self.config.intrinsic_water_red_deficit_margin
            )
            water_chroma_residual = torch.maximum(blue_residual, 0.75 * cyan_residual)
            residual_chroma_weight = torch.maximum(
                water_probabilities["residual_water"],
                water_probabilities["connected_freewater"] * 0.5,
            )
            residual_water_chroma_loss = self._weighted_mean(water_chroma_residual, residual_chroma_weight)

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
            "water_empty_occupancy_loss": water_empty_occupancy_loss * self.config.water_empty_loss_weight,
            "water_transition_alpha_loss": water_transition_alpha_loss * self.config.water_transition_alpha_loss_weight,
            "water_intrinsic_null_loss": water_intrinsic_null_loss * self.config.water_intrinsic_null_loss_weight,
            "water_background_ownership_loss": water_background_ownership_loss * self.config.water_background_ownership_loss_weight,
            "residual_water_chroma_loss": residual_water_chroma_loss * self.config.residual_water_chroma_loss_weight,
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
        gt_rgb_image = gt_rgb
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

        intrinsic_raw = outputs.get("intrinsic_color_render_raw", outputs["intrinsic_color_render"])
        intrinsic_clean = outputs["intrinsic_color_render"]
        images_dict['intrinsic_color_render_raw'] = intrinsic_raw
        images_dict['rendered_underwater_image'] = outputs["rgb"]  
         
        images_dict['water_background_image'] = outputs["water_background_image"]   
        images_dict['background_backscatter_coefficients'] = outputs["background_backscatter_coefficients"]   
        images_dict['background_attenuation_coefficients'] = outputs["background_attenuation_coefficients"]            
            
        images_dict['depth_GT'] = gt_depth
        if self.config.enable_water_null_occupancy:
            eval_water_probabilities = self._build_water_null_probabilities(
                gt_depth.to(self.device),
                gt_rgb_image,
                outputs["water_background_image"],
                intrinsic_raw,
                outputs["accumulation"],
            )
            eval_water_probability = eval_water_probabilities["null"]
            eval_pure_water_probability = eval_water_probabilities["pure"]
            eval_transition_water_probability = eval_water_probabilities["transition"]
            eval_strict_pure_probability = self._build_strict_water_null_probability(
                eval_pure_water_probability,
                self.config.water_null_strict_threshold,
                self.config.water_pure_max_fraction,
            )
            eval_strict_transition_probability = self._build_strict_water_null_probability(
                eval_transition_water_probability,
                self.config.water_transition_strict_threshold,
                self.config.water_transition_max_fraction,
            )
            eval_strict_water_probability = torch.maximum(
                eval_strict_pure_probability,
                eval_strict_transition_probability,
            )
            eval_alpha_water = torch.maximum(
                outputs.get("water_accumulation", torch.zeros_like(outputs["accumulation"])),
                outputs["accumulation"] * eval_strict_pure_probability,
            )
            eval_intrinsic_visibility = self._build_intrinsic_visibility_gate(
                outputs["accumulation"],
                eval_water_probability,
                eval_alpha_water,
                eval_water_probabilities["scene_rescue"],
                eval_water_probabilities["residual_water"],
                eval_water_probabilities["connected_freewater"],
            )
            intrinsic_clean = intrinsic_raw * eval_intrinsic_visibility
            images_dict['intrinsic_visibility'] = eval_intrinsic_visibility
            images_dict['water_pure_probability'] = eval_pure_water_probability
            images_dict['water_transition_probability'] = eval_transition_water_probability
            images_dict['water_null_probability'] = eval_water_probability
            images_dict['water_null_strict_probability'] = eval_strict_water_probability
            images_dict['water_horizon_probability'] = eval_water_probabilities["horizon"]
            images_dict['depth_edge_support'] = eval_water_probabilities["depth_edge_support"]
            images_dict['color_residual_support'] = eval_water_probabilities["color_residual_support"]
            images_dict['intrinsic_scene_rescue'] = eval_water_probabilities["scene_rescue"]
            images_dict['residual_water_probability'] = eval_water_probabilities["residual_water"]
            images_dict['connected_freewater_probability'] = eval_water_probabilities["connected_freewater"]
        images_dict['intrinsic_color_render'] = intrinsic_clean
        
        depth_ED = outputs["depth"]
        q99 = torch.quantile(depth_ED.flatten(), 0.99)
        depth_ED_clipped = torch.clamp(depth_ED, max=q99)
        depth_ED_normalized = depth_ED_clipped / q99
        images_dict['depth_ED'] = depth_ED_normalized

            
        return metrics_dict, images_dict
