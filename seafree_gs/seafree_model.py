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
from seafree_gs.aquanull3d import SurfaceCarrierField
from seafree_gs.path_integrated_water import CompactHeterogeneousWaterField, PathIntegratedWaterRenderer
from seafree_gs.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases

from torchmetrics.functional.regression import pearson_corrcoef


class WaterFormationCalibrator(nn.Module):
    """Low-frequency, direction-conditioned calibration for the water formation model.

    Unlike the removed Underwater Fidelity Adapter, this module never edits the
    final rendered RGB image in 2D image space. It predicts small residuals for
    ambient light and log-scale corrections for backscatter / attenuation before
    the underwater image is formed. The final layer is zero-initialized so old
    checkpoints keep the original SeaFree-GS water model behavior.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        final_conv = self.net[-1]
        if isinstance(final_conv, nn.Linear):
            nn.init.zeros_(final_conv.weight)
            nn.init.zeros_(final_conv.bias)

    def forward(self, encoded_directions: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(encoded_directions.float()))


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
    """Propagate high-confidence border water while the 3D carrier protects surfaces."""
    connected_freewater_iterations: int = 24
    """Number of 3x3 soft dilation steps for the border-connected freewater prior."""
    connected_freewater_strength: float = 0.85
    """Strength with which connected freewater evidence contributes to null probability."""
    connected_freewater_visibility_strength: float = 1.00
    """Strength with which connected freewater evidence suppresses final intrinsic visibility."""
    connected_freewater_visibility_gamma: float = 1.40
    """Nonlinear visibility hardening for border-connected freewater sheets."""
    enable_physics_anchored_water_matte: bool = True
    """Use a physics/color-anchor water matte that does not require border connectivity."""
    water_anchor_color_sigma: float = 0.13
    """Color-distance scale for matching pixels to the image-level water color anchor."""
    water_anchor_chroma_margin: float = 0.015
    """Green/blue excess over red that indicates water-colored haze in the input image."""
    water_anchor_chroma_scale: float = 0.060
    """Softness of the input-image water chroma detector."""
    water_anchor_matte_strength: float = 1.20
    """Strength of the anchored freewater matte inside the final responsibility split."""
    water_anchor_spatial_iterations: int = 4
    """Number of support-constrained spatial completion steps for non-border water sheets."""
    water_anchor_spatial_strength: float = 0.55
    """How much the completed matte fills smooth water-like holes and bands."""
    water_anchor_scene_protection: float = 0.90
    """How strongly scene evidence suppresses the anchored freewater matte."""
    water_anchor_low_alpha_strength: float = 0.35
    """How much low accumulated alpha can boost freewater when color/depth agree."""
    enable_layered_veil_scene_factorization: bool = True
    """Factor clean rendering into empty freewater, attached water veil, scene core, and uncertainty layers."""
    layer_scene_core_water_suppression: float = 0.90
    """How strongly water-colored context suppresses raw-intrinsic scene rescue."""
    layer_empty_freewater_strength: float = 1.15
    """Strength of the strict empty-freewater layer used for black background."""
    layer_attached_veil_strength: float = 1.65
    """Strength of the object-attached veil layer used for de-veiling rather than blacking out."""
    layer_attached_veil_max_alpha: float = 0.60
    """Maximum veil alpha removed from raw intrinsic colors in attached-veil regions."""
    layer_uncertain_deveil_strength: float = 0.30
    """Conservative de-veiling strength in uncertain water-colored boundary regions."""
    layer_deveil_darkening_limit: float = 0.45
    """Maximum fractional luminance darkening allowed for attached-veil de-veiling before freewater blacking."""
    layer_empty_freewater_strict_threshold: float = 0.62
    """Strict threshold for destructive black-background losses on empty freewater only."""
    layer_empty_black_loss_weight: float = 0.045
    """Loss weight that drives high-confidence empty freewater to black in clean rendering."""
    layer_attached_chroma_loss_weight: float = 0.030
    """Loss weight that removes blue/cyan residuals from attached water veil without deleting structure."""
    layer_detail_preservation_loss_weight: float = 0.012
    """Loss weight that keeps de-veiled attached objects from becoming over-smoothed."""
    layer_veil_smoothness_loss_weight: float = 0.004
    """Loss weight that smooths veil alpha away from strong scene boundaries to reduce bands."""
    layer_exclusivity_loss_weight: float = 0.006
    """Loss weight discouraging empty-freewater, attached-veil, and scene-core layers from all being high."""
    enable_water_scene_responsibility_field: bool = True
    """Use a cross-view freewater / residual-haze / scene responsibility field."""
    responsibility_temperature: float = 0.18
    """Soft assignment temperature for three-way water-scene responsibility normalization."""
    responsibility_evidence_floor: float = 0.025
    """Small background evidence that keeps low-confidence pixels uncertain rather than over-classified."""
    responsibility_freewater_strength: float = 1.00
    """Strength with which freewater responsibility contributes to clean null visibility."""
    responsibility_haze_strength: float = 0.70
    """Strength with which residual-haze responsibility suppresses only water-colored clean residuals."""
    responsibility_scene_strength: float = 1.15
    """Strength with which scene responsibility protects weak distant geometry and color-card regions."""
    responsibility_uncertainty_visibility_floor: float = 0.18
    """Minimum clean visibility retained for uncertain pixels to avoid hard false-positive blackouts."""
    responsibility_freewater_visibility_gamma: float = 2.20
    """Nonlinear clean visibility hardening for high-confidence freewater only."""
    responsibility_haze_visibility_gamma: float = 1.20
    """Softer clean visibility suppression for residual haze over real objects."""
    responsibility_strict_freewater_threshold: float = 0.72
    """Minimum freewater responsibility before destructive empty-space losses are applied."""
    responsibility_strict_haze_threshold: float = 0.58
    """Minimum residual-haze responsibility for non-destructive chroma / fidelity losses."""
    responsibility_freewater_max_fraction: float = 0.08
    """Maximum per-image fraction allowed to receive destructive freewater losses."""
    responsibility_haze_max_fraction: float = 0.22
    """Maximum per-image fraction allowed to receive residual-haze losses."""
    responsibility_scene_fidelity_loss_weight: float = 0.035
    """Extra underwater reconstruction weight on high-confidence scene responsibility pixels."""
    responsibility_haze_fidelity_loss_weight: float = 0.018
    """Extra underwater reconstruction weight on residual-haze pixels without deleting structure."""
    responsibility_boundary_fidelity_loss_weight: float = 0.008
    """Gradient reconstruction weight around responsibility transitions to reduce dark halos."""
    enable_water_formation_calibrator: bool = True
    """Use a direction-conditioned physical water formation calibrator instead of a 2D RGB adapter."""
    water_calibrator_hidden_dim: int = 32
    """Hidden width of the water formation calibration MLP."""
    water_calibrator_ambient_delta: float = 0.050
    """Maximum additive ambient-light correction from the water calibrator."""
    water_calibrator_coeff_log_scale: float = 0.32
    """Maximum log-scale correction for backscatter and attenuation coefficients."""
    water_calibrator_l2_weight: float = 0.0006
    """Regularization weight that keeps physical water calibration residuals small."""
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

    enable_aquanull3d: bool = True
    """Use persistent multi-view 3D surface-carrier certification."""
    carrier_update_every: int = 100
    """Update carrier evidence every N training iterations."""
    carrier_min_distinct_views: int = 6
    """Minimum distinct training cameras in one carrier evidence window."""
    carrier_warmup_steps: int = 4000
    """Keep exact AquaNull-LVSF geometry while WPP and GS appearance stabilize."""
    carrier_epoch_steps: int = 0
    """Angular coverage window length; zero derives it from camera count."""
    carrier_min_certified_epochs: int = 2
    """Consecutive decisive angular observations required to commit ownership."""
    carrier_chunk_size: int = 262144
    """Maximum number of projected Gaussians processed by one carrier update chunk."""
    carrier_growth_start_step: int = 4000
    """Start blocking growth of repeatedly observed non-surface Gaussians."""
    carrier_prune_start_step: int = 16000
    """Start conservative carrier pruning after normal densification has stopped."""
    carrier_prune_max_fraction: float = 0.0025
    """Maximum fraction removed by carrier pruning at one refinement iteration."""

    enable_path_integrated_transport: bool = True
    """Use the bounded scene-global field and analytic per-Gaussian transport."""
    water_field_resolution: int = 16
    water_query_downscale: int = 8
    water_prefix_depth_bins: int = 8
    """Number of front-to-back segments in the low-resolution frustum prefix."""
    water_transport_depth_scale: float = 10.0
    water_transport_far_distance: float = 4.0
    water_field_start_step: int = 15000
    """Start learning spatial transport after normal GS densification."""
    water_field_freeze_step: int = 23000
    """Freeze the medium field and leave the final iterations to GS stabilization."""
    water_field_update_every: int = 4
    """Update only SAFT on one step in this block-coordinate interval."""
    water_support_update_every: int = 500
    """Refresh the GPU free-space support grid from carrier posteriors."""
    water_field_ambient_residual_scale: float = 0.04
    """Maximum low-rank spatial ambient residual around the WPP."""
    water_field_coefficient_log_scale: float = 0.20
    """Maximum log residual for low-rank attenuation/backscatter correction."""
    
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
        metadata = kwargs.get("metadata") or {}
        sfm_error = metadata.get("points3D_error")
        sfm_tracks = metadata.get("points3D_num_points2D")
        self.seed_sfm_reliability: Optional[torch.Tensor] = None
        if (
            seed_points is not None
            and isinstance(sfm_error, torch.Tensor)
            and isinstance(sfm_tracks, torch.Tensor)
            and sfm_error.numel() == seed_points[0].shape[0]
            and sfm_tracks.numel() == seed_points[0].shape[0]
        ):
            error = sfm_error.detach().float().clamp_min(0.0)
            tracks = sfm_tracks.detach().float().clamp_min(0.0)
            positive_error = error[error > 0]
            error_scale = (
                positive_error.median() if positive_error.numel() > 0 else error.new_tensor(1.0)
            ).clamp_min(1e-4)
            track_support = tracks / (tracks + 4.0)
            reprojection_support = torch.exp(-error / (2.0 * error_scale))
            self.seed_sfm_reliability = (track_support * reprojection_support).clamp(0.0, 1.0)
        self.max_depth = max_depth * 4
        super().__init__(*args, **kwargs)

    def _ensure_responsibility_config_defaults(self):
        """Populate new-module defaults when loading older saved configs."""
        defaults = {
            "enable_physics_anchored_water_matte": True,
            "water_anchor_color_sigma": 0.13,
            "water_anchor_chroma_margin": 0.015,
            "water_anchor_chroma_scale": 0.060,
            "water_anchor_matte_strength": 1.20,
            "water_anchor_spatial_iterations": 4,
            "water_anchor_spatial_strength": 0.55,
            "water_anchor_scene_protection": 0.90,
            "water_anchor_low_alpha_strength": 0.35,
            "enable_layered_veil_scene_factorization": True,
            "enable_connected_freewater_prior": True,
            "layer_scene_core_water_suppression": 0.90,
            "layer_empty_freewater_strength": 1.15,
            "layer_attached_veil_strength": 1.65,
            "layer_attached_veil_max_alpha": 0.60,
            "layer_uncertain_deveil_strength": 0.30,
            "layer_deveil_darkening_limit": 0.45,
            "layer_empty_freewater_strict_threshold": 0.62,
            "layer_empty_black_loss_weight": 0.045,
            "layer_attached_chroma_loss_weight": 0.030,
            "layer_detail_preservation_loss_weight": 0.012,
            "layer_veil_smoothness_loss_weight": 0.004,
            "layer_exclusivity_loss_weight": 0.006,
            "enable_water_scene_responsibility_field": True,
            "responsibility_temperature": 0.18,
            "responsibility_evidence_floor": 0.025,
            "responsibility_freewater_strength": 1.00,
            "responsibility_haze_strength": 0.70,
            "responsibility_scene_strength": 1.15,
            "responsibility_uncertainty_visibility_floor": 0.18,
            "responsibility_freewater_visibility_gamma": 2.20,
            "responsibility_haze_visibility_gamma": 1.20,
            "responsibility_strict_freewater_threshold": 0.72,
            "responsibility_strict_haze_threshold": 0.58,
            "responsibility_freewater_max_fraction": 0.08,
            "responsibility_haze_max_fraction": 0.22,
            "responsibility_scene_fidelity_loss_weight": 0.035,
            "responsibility_haze_fidelity_loss_weight": 0.018,
            "responsibility_boundary_fidelity_loss_weight": 0.008,
            "enable_water_formation_calibrator": True,
            "water_calibrator_hidden_dim": 32,
            "water_calibrator_ambient_delta": 0.050,
            "water_calibrator_coeff_log_scale": 0.32,
            "water_calibrator_l2_weight": 0.0006,
            "enable_aquanull3d": True,
            "carrier_update_every": 100,
            "carrier_min_distinct_views": 6,
            "carrier_warmup_steps": 4000,
            "carrier_epoch_steps": 0,
            "carrier_min_certified_epochs": 2,
            "carrier_chunk_size": 262144,
            "carrier_growth_start_step": 4000,
            "carrier_prune_start_step": 16000,
            "carrier_prune_max_fraction": 0.0025,
            "enable_path_integrated_transport": True,
            "water_field_resolution": 16,
            "water_query_downscale": 8,
            "water_prefix_depth_bins": 8,
            "water_transport_depth_scale": 10.0,
            "water_transport_far_distance": 4.0,
            "water_field_start_step": 15000,
            "water_field_freeze_step": 23000,
            "water_field_update_every": 4,
            "water_support_update_every": 500,
            "water_field_ambient_residual_scale": 0.04,
            "water_field_coefficient_log_scale": 0.20,
        }
        for name, value in defaults.items():
            if not hasattr(self.config, name):
                setattr(self.config, name, value)

    def _uses_nextgen_water_model(self) -> bool:
        # The standalone nextgen branch replaced the validated LVSF pipeline and
        # caused the v1/v2 regressions. New modules are integrated into LVSF.
        return False

    def _uses_surface_carrier(self) -> bool:
        return bool(self.config.enable_aquanull3d)

    def _uses_legacy_water_lifecycle(self) -> bool:
        return (
            not self._uses_surface_carrier()
            or self.config.enable_water_lifecycle_growth_suppression
            or self.config.enable_water_lifecycle_pruning
        )

    def populate_modules(self):
        self._ensure_responsibility_config_defaults()

        
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
        self.water_formation_calibrator = WaterFormationCalibrator(
            in_dim=mlp_dim_in,
            hidden_dim=max(4, int(self.config.water_calibrator_hidden_dim)),
        )
        
            
        
        
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
        if self._uses_legacy_water_lifecycle():
            self.strategy_state["water_exposure_score"] = torch.zeros(num_points, device=means.device)
            self.strategy_state["scene_exposure_score"] = torch.zeros(num_points, device=means.device)
            self.strategy_state["depth_edge_support_score"] = torch.zeros(num_points, device=means.device)
            self.strategy_state["color_residual_support_score"] = torch.zeros(num_points, device=means.device)
            self.strategy_state["view_consistency_score"] = torch.zeros(num_points, device=means.device)

        self.surface_carrier = SurfaceCarrierField(
            num_points=num_points,
            device=means.device,
            update_every=self.config.carrier_update_every,
            min_distinct_views=self.config.carrier_min_distinct_views,
            num_train_views=self.num_train_data,
            warmup_steps=self.config.carrier_warmup_steps,
            epoch_steps=self.config.carrier_epoch_steps,
            min_certified_epochs=self.config.carrier_min_certified_epochs,
            depth_scale=self.config.water_transport_depth_scale,
            chunk_size=self.config.carrier_chunk_size,
            sfm_reliability=self.seed_sfm_reliability,
        )
        water_field = CompactHeterogeneousWaterField(
            resolution=self.config.water_field_resolution,
            ambient_residual_scale=self.config.water_field_ambient_residual_scale,
            coefficient_log_scale=self.config.water_field_coefficient_log_scale,
        )
        # SceneBox is already robustly normalized by the dataparser and avoids
        # an additional CPU pass over the full COLMAP cloud.
        water_field.set_bounds(self.scene_box.aabb.to(means))
        self.path_integrated_renderer = PathIntegratedWaterRenderer(
            field=water_field,
            query_downscale=self.config.water_query_downscale,
            depth_scale=self.config.water_transport_depth_scale,
            far_distance=self.config.water_transport_far_distance,
            num_depth_bins=self.config.water_prefix_depth_bins,
            query_chunk_size=self.config.carrier_chunk_size,
        )
        if self._uses_surface_carrier():
            self.surface_carrier.bind_strategy_state(self.strategy_state)
        
        self.foreground_mask_cache = {}
        legacy_state_size = num_points if self._uses_legacy_water_lifecycle() else 0
        self.register_buffer("water_exposure_score", torch.zeros(legacy_state_size), persistent=False)
        self.register_buffer("scene_exposure_score", torch.zeros(legacy_state_size), persistent=False)
        self.register_buffer("depth_edge_support_score", torch.zeros(legacy_state_size), persistent=False)
        self.register_buffer("color_residual_support_score", torch.zeros(legacy_state_size), persistent=False)
        self.register_buffer("view_consistency_score", torch.zeros(legacy_state_size), persistent=False)
        self.last_water_probability_mean = torch.tensor(0.0)
        self.last_water_pure_probability_mean = torch.tensor(0.0)
        self.last_water_transition_probability_mean = torch.tensor(0.0)
        self.last_water_strict_probability_mean = torch.tensor(0.0)
        self.last_water_horizon_probability_mean = torch.tensor(0.0)
        self.last_residual_water_probability_mean = torch.tensor(0.0)
        self.last_connected_freewater_probability_mean = torch.tensor(0.0)
        self.last_responsibility_freewater_mean = torch.tensor(0.0)
        self.last_responsibility_haze_mean = torch.tensor(0.0)
        self.last_responsibility_scene_mean = torch.tensor(0.0)
        self.last_responsibility_uncertainty_mean = torch.tensor(0.0)
        self.last_anchored_freewater_probability_mean = torch.tensor(0.0)
        self.last_anchored_haze_probability_mean = torch.tensor(0.0)
        self.last_water_anchor_similarity_mean = torch.tensor(0.0)
        self.last_water_matte_scene_protection_mean = torch.tensor(0.0)
        self.last_empty_freewater_probability_mean = torch.tensor(0.0)
        self.last_attached_veil_probability_mean = torch.tensor(0.0)
        self.last_scene_core_probability_mean = torch.tensor(0.0)
        self.last_veil_alpha_mean = torch.tensor(0.0)
        self.last_water_calibration_delta_mean = torch.tensor(0.0)
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
        dict = dict.copy()
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
        self.surface_carrier.resize(newp, self.gauss_params["means"].device)
        strict = kwargs.pop("strict", True)
        if self.config.enable_water_formation_calibrator:
            calibrator_keys = {
                f"water_formation_calibrator.{name}"
                for name, _ in self.water_formation_calibrator.named_parameters()
            }
            if any(key not in dict for key in calibrator_keys):
                strict = False
        nextgen_keys = {
            key
            for key in self.state_dict().keys()
            if key.startswith("surface_carrier.")
            or key.startswith("path_integrated_renderer.")
        }
        has_water_field_bounds = all(
            key in dict
            for key in (
                "path_integrated_renderer.field.aabb_min",
                "path_integrated_renderer.field.aabb_max",
            )
        )
        carrier_is_redesigned = all(
            key in dict
            for key in (
                "surface_carrier.medium_evidence",
                "surface_carrier.certified_state",
                "surface_carrier.lifetime_view_mask",
                "surface_carrier.window_medium_evidence",
                "surface_carrier.lifetime_coverage",
                "surface_carrier.direction_sum",
                "surface_carrier.window_view_ids",
                "surface_carrier.topology_medium_support",
                "surface_carrier.topology_surface_support",
            )
        )
        transport_is_redesigned = all(
            key in dict
            for key in (
                "path_integrated_renderer.field.volume",
                "path_integrated_renderer.field.latent_running_mean",
                "path_integrated_renderer.field.free_space_support",
                "path_integrated_renderer.field.support_observation_mass",
                "path_integrated_renderer.field.medium_optical_mass",
                "path_integrated_renderer.field.surface_endpoint_mass",
            )
        )
        if not carrier_is_redesigned:
            for key in [key for key in dict if key.startswith("surface_carrier.")]:
                dict.pop(key)
            strict = False
        if not transport_is_redesigned:
            for key in [
                key for key in dict if key.startswith("path_integrated_renderer.")
            ]:
                dict.pop(key)
            has_water_field_bounds = False
            strict = False
        if any(key not in dict for key in nextgen_keys):
            strict = False
        current_state = self.state_dict()
        for key in list(dict.keys()):
            if not (
                key.startswith("surface_carrier.")
                or key.startswith("path_integrated_renderer.")
            ):
                continue
            if key in current_state and current_state[key].shape != dict[key].shape:
                # The redesigned modules intentionally changed persistent state.
                # Older v1 checkpoints still load their GS parameters while the
                # incompatible carrier/medium state is reinitialized safely.
                dict.pop(key)
                strict = False
        super().load_state_dict(dict, strict=strict, **kwargs)
        if not has_water_field_bounds:
            self.path_integrated_renderer.field.set_bounds(self.means)
        if self._uses_surface_carrier():
            self.surface_carrier.bind_strategy_state(self.strategy_state)

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def step_post_backward(self, step):
        assert step == self.step
        if getattr(self, "_saft_field_update_step", False):
            # No Gaussian parameter participates in a dedicated field step, so
            # gsplat has no valid screen-space gradient to accumulate.
            return
        no_grow_mask = None
        prune_mask = None
        if self.config.enable_water_null_occupancy:
            if self.config.enable_water_lifecycle_growth_suppression:
                self._suppress_water_dominated_growth()
                no_grow_mask = self._get_water_lifecycle_no_grow_mask()
            if self.config.enable_water_lifecycle_pruning:
                prune_mask = self._get_water_lifecycle_prune_mask()

        if self._uses_surface_carrier() and step % self.config.refine_every == 0:
            self.surface_carrier.sync_from_strategy_state(self.strategy_state)
            carrier_no_grow, carrier_prune = self.surface_carrier.lifecycle_masks(
                step=step,
                grow_start=self.config.carrier_growth_start_step,
                prune_start=self.config.carrier_prune_start_step,
                max_prune_fraction=self.config.carrier_prune_max_fraction,
            )
            if carrier_no_grow is not None:
                no_grow_mask = (
                    carrier_no_grow
                    if no_grow_mask is None
                    else torch.logical_or(no_grow_mask, carrier_no_grow)
                )
            if carrier_prune is not None:
                prune_mask = (
                    carrier_prune
                    if prune_mask is None
                    else torch.logical_or(prune_mask, carrier_prune)
                )

        if prune_mask is not None and bool(prune_mask.any().item()):
            # External pruning runs before growth. A no-grow mask still has the
            # pre-prune length and therefore cannot be applied in the same pass.
            no_grow_mask = None
        if hasattr(self.strategy, "set_external_no_grow_mask"):
            self.strategy.set_external_no_grow_mask(no_grow_mask)
        if hasattr(self.strategy, "set_external_prune_mask"):
            self.strategy.set_external_prune_mask(prune_mask)
        self.last_water_no_grow_candidates = (
            int(no_grow_mask.sum().item()) if no_grow_mask is not None else 0
        )
        self.last_water_prune_candidates = (
            int(prune_mask.sum().item()) if prune_mask is not None else 0
        )
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
        elif self._uses_surface_carrier():
            self.surface_carrier.sync_from_strategy_state(self.strategy_state)

    def _prepare_gaussian_densification_backward(self) -> None:
        """Retain projected-mean gradients only when Gaussians are trainable."""
        if not self.training or getattr(self, "_saft_field_update_step", False):
            return
        self.strategy.step_pre_backward(
            self.gauss_params,
            self.optimizers,
            self.strategy_state,
            self.step,
            self.info,
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
        field_window = (
            self.config.enable_path_integrated_transport
            and self.config.water_field_start_step <= step < self.config.water_field_freeze_step
        )
        field_interval = max(int(self.config.water_field_update_every), 1)
        field_trainable = field_window and (
            (step - self.config.water_field_start_step) % field_interval == 0
        )
        self._saft_field_update_step = field_trainable
        for parameter in self.path_integrated_renderer.field.parameters():
            parameter.requires_grad_(field_trainable)

        # On SAFT steps the spatial transport receives an identifiable target:
        # Gaussian appearance/opacity and the directional WPP cannot move to
        # explain the same residual. Normal steps freeze SAFT and optimize them.
        appearance_trainable = not field_trainable
        for parameter in self.gauss_params.values():
            parameter.requires_grad_(appearance_trainable)
        identifiable_modules = [
            self.line_of_sight_direction_encoding,
            self.water_properties_predictor,
            self.water_formation_calibrator,
        ]
        if hasattr(self, "camera_optimizer"):
            identifiable_modules.append(self.camera_optimizer)
        if hasattr(self, "bil_grids"):
            identifiable_modules.append(self.bil_grids)
        for module in identifiable_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(appearance_trainable)

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

        water_properties_params = list(self.water_properties_predictor.parameters())
        if self.config.enable_water_formation_calibrator:
            water_properties_params += list(self.water_formation_calibrator.parameters())
        gps["water_properties_predictor"] = water_properties_params
        if self.config.enable_path_integrated_transport:
            gps["water_transport_field"] = list(
                self.path_integrated_renderer.field.parameters()
            )
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
        if (
            not self.config.enable_connected_freewater_prior
            or seed.numel() == 0
        ):
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

    def _support_guided_spatial_completion(
        self,
        matte: torch.Tensor,
        support: torch.Tensor,
        iterations: int,
        strength: float,
    ) -> torch.Tensor:
        """Fill local holes in water-like regions without requiring image-border connectivity."""
        if matte.numel() == 0 or iterations <= 0 or strength <= 0:
            return matte.clamp(0.0, 1.0)
        matte_2d = matte.detach().squeeze(-1).clamp(0.0, 1.0)
        support_2d = support.detach().squeeze(-1).clamp(0.0, 1.0)
        completed = matte_2d[None, None, ...]
        support_chw = support_2d[None, None, ...]
        for _ in range(max(0, int(iterations))):
            expanded = F.max_pool2d(completed, kernel_size=3, stride=1, padding=1)
            completed = torch.maximum(completed, expanded * support_chw)
        completed = completed.squeeze(0).permute(1, 2, 0)
        return torch.maximum(matte, completed * max(min(strength, 1.0), 0.0)).clamp(0.0, 1.0)

    def _build_physics_anchored_water_matte(
        self,
        depth_confidence: torch.Tensor,
        transition_depth_confidence: torch.Tensor,
        texture_confidence: torch.Tensor,
        intrinsic_texture_confidence: torch.Tensor,
        low_depth_edge_confidence: torch.Tensor,
        edge_protection: torch.Tensor,
        scene_rescue: torch.Tensor,
        depth_edge_support: torch.Tensor,
        residual_water_probability: torch.Tensor,
        gt_underwater_image: torch.Tensor,
        water_background_image: torch.Tensor,
        intrinsic_image: Optional[torch.Tensor],
        accumulation: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Build a physics/color anchored water matte.

        The previous topology matte failed when pure-water seeds were not
        connected to the image border. This matte estimates an image-level water
        color anchor from far, smooth, low-scene pixels and then combines anchor
        similarity with depth / texture / raw-intrinsic scene counter-evidence.
        High-confidence freewater can become black; haze remains non-destructive.
        """
        reference = depth_confidence[..., :1]
        zero = torch.zeros_like(reference)
        if not self.config.enable_physics_anchored_water_matte:
            return {
                "freewater": zero,
                "haze": residual_water_probability.detach().clamp(0.0, 1.0),
                "support": zero,
                "anchor_similarity": zero,
                "input_chroma": zero,
                "scene_protection": scene_rescue.detach().clamp(0.0, 1.0),
                "low_structure": zero,
            }

        rgb = torch.nan_to_num(
            gt_underwater_image[..., :3].detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        water_bg = torch.nan_to_num(
            water_background_image[..., :3].detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        height, width = reference.shape[:2]
        border_boost = torch.zeros_like(reference)
        top_rows = max(1, int(height * 0.18))
        border_boost[:top_rows, :, :] = 1.0
        border_boost[0, :, :] = 1.0
        border_boost[-1, :, :] = 1.0
        border_boost[:, 0, :] = 1.0
        border_boost[:, -1, :] = 1.0

        scene_guard = (1.0 - 0.75 * scene_rescue.detach().clamp(0.0, 1.0)).clamp(0.05, 1.0)
        anchor_weight = (
            transition_depth_confidence.detach().clamp(0.0, 1.0)
            * texture_confidence.detach().clamp(0.0, 1.0)
            * low_depth_edge_confidence.detach().clamp(0.0, 1.0)
            * edge_protection.detach().clamp(0.05, 1.0)
            * scene_guard
            * (1.0 + 0.50 * border_boost)
        ).clamp(0.0, 1.0)
        anchor_denom = anchor_weight.sum().clamp_min(1e-6)
        anchor_color = (rgb * anchor_weight).sum(dim=(0, 1), keepdim=True) / anchor_denom
        fallback_weight = (
            transition_depth_confidence.detach().clamp(0.0, 1.0)
            * texture_confidence.detach().clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        fallback_denom = fallback_weight.sum().clamp_min(1e-6)
        fallback_color = (water_bg * fallback_weight).sum(dim=(0, 1), keepdim=True) / fallback_denom
        anchor_color = torch.where(
            (anchor_weight.sum() > 1e-4).view(1, 1, 1),
            anchor_color,
            fallback_color,
        ).detach()

        sigma = max(float(self.config.water_anchor_color_sigma), 1e-4)
        anchor_distance = torch.abs(rgb - anchor_color).mean(dim=-1, keepdim=True)
        wpp_distance = torch.abs(rgb - water_bg).mean(dim=-1, keepdim=True)
        anchor_similarity = torch.maximum(
            torch.exp(-anchor_distance / sigma),
            0.75 * torch.exp(-wpp_distance / sigma),
        ).clamp(0.0, 1.0)

        red = rgb[..., 0:1]
        green = rgb[..., 1:2]
        blue = rgb[..., 2:3]
        blue_green = torch.maximum(green, blue)
        chroma_margin = float(self.config.water_anchor_chroma_margin)
        chroma_scale = max(float(self.config.water_anchor_chroma_scale), 1e-4)
        blue_green_chroma = torch.sigmoid((blue_green - red - chroma_margin) / chroma_scale * 4.0)
        cyan_chroma = torch.sigmoid((torch.minimum(green, blue) - red - chroma_margin) / chroma_scale * 4.0)
        input_chroma = torch.maximum(blue_green_chroma, 0.75 * cyan_chroma).clamp(0.0, 1.0)

        texture_scene = (1.0 - texture_confidence.detach().clamp(0.0, 1.0)).clamp(0.0, 1.0)
        intrinsic_texture_scene = (1.0 - intrinsic_texture_confidence.detach().clamp(0.0, 1.0)).clamp(0.0, 1.0)
        nonblue_scene = torch.sigmoid(
            (torch.maximum(red, green) - 0.92 * blue)
            / max(chroma_scale, 1e-4)
            * 3.0
        )
        # Green/blue water often satisfies a naive "non-blue" test because the
        # green channel is close to or above blue.  Cancel this scene cue when
        # the input itself has strong water-colored chroma.
        nonblue_scene = nonblue_scene * (1.0 - 0.70 * input_chroma).clamp(0.0, 1.0)
        anchor_deviation_scene = 1.0 - torch.exp(-anchor_distance / max(sigma * 1.50, 1e-4))
        # Gaussian accumulation is visibility, not independent evidence that a
        # carrier is a surface. Using it here made fog splats protect themselves.
        del accumulation

        intrinsic_nonblue_scene = torch.zeros_like(reference)
        intrinsic_luma_scene = torch.zeros_like(reference)
        if intrinsic_image is not None and intrinsic_image.numel() > 0:
            intrinsic = torch.nan_to_num(
                intrinsic_image[..., :3].detach(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            intrinsic_red_green = torch.maximum(intrinsic[..., 0:1], intrinsic[..., 1:2])
            intrinsic_nonblue_scene = torch.sigmoid(
                (intrinsic_red_green - 0.90 * intrinsic[..., 2:3])
                / max(chroma_scale, 1e-4)
                * 3.0
            )
            luma_threshold = max(self.config.intrinsic_rescue_luma_threshold, 1e-4)
            intrinsic_luma = intrinsic.mean(dim=-1, keepdim=True)
            intrinsic_luma_scene = torch.sigmoid((intrinsic_luma - luma_threshold) / luma_threshold * 4.0)

        color_scene = anchor_deviation_scene * torch.maximum(
            nonblue_scene,
            torch.maximum(texture_scene, intrinsic_nonblue_scene * intrinsic_luma_scene),
        )
        structure_scene = torch.maximum(
            depth_edge_support.detach().clamp(0.0, 1.0),
            torch.maximum(texture_scene, intrinsic_texture_scene),
        )
        scene_protection = torch.maximum(
            scene_rescue.detach().clamp(0.0, 1.0),
            torch.maximum(0.70 * structure_scene, 0.85 * color_scene),
        )
        scene_protection = scene_protection.clamp(0.0, 1.0)

        low_structure = torch.sqrt(
            (
                texture_confidence.detach().clamp(0.0, 1.0)
                * intrinsic_texture_confidence.detach().clamp(0.0, 1.0)
                * low_depth_edge_confidence.detach().clamp(0.0, 1.0)
            ).clamp_min(0.0)
        )
        water_color_evidence = torch.maximum(anchor_similarity, 0.70 * input_chroma).clamp(0.0, 1.0)
        water_context = (
            transition_depth_confidence.detach().clamp(0.0, 1.0)
            * low_structure
            * water_color_evidence
        ).clamp(0.0, 1.0)
        scene_protection = (
            scene_protection
            * (1.0 - 0.65 * water_context).clamp(0.12, 1.0)
        ).clamp(0.0, 1.0)
        matte_scene_guard = (
            1.0 - float(self.config.water_anchor_scene_protection) * scene_protection
        ).clamp(0.03, 1.0)
        freewater = (
            transition_depth_confidence.detach().clamp(0.0, 1.0)
            * low_structure
            * water_color_evidence
            * matte_scene_guard
        )
        deep_freewater = (
            depth_confidence.detach().clamp(0.0, 1.0)
            * texture_confidence.detach().clamp(0.0, 1.0)
            * anchor_similarity
            * matte_scene_guard
        )
        freewater = torch.maximum(freewater, deep_freewater)

        spatial_support = (
            transition_depth_confidence.detach().clamp(0.0, 1.0)
            * low_structure
            * water_color_evidence
            * (1.0 - 0.75 * scene_protection).clamp(0.05, 1.0)
        ).clamp(0.0, 1.0)
        freewater = self._support_guided_spatial_completion(
            freewater.clamp(0.0, 1.0),
            spatial_support,
            int(self.config.water_anchor_spatial_iterations),
            float(self.config.water_anchor_spatial_strength),
        )

        haze = torch.maximum(
            residual_water_probability.detach().clamp(0.0, 1.0),
            transition_depth_confidence.detach().clamp(0.0, 1.0) * input_chroma * anchor_similarity,
        )
        haze = haze * (0.25 + 0.75 * scene_protection).clamp(0.0, 1.0)
        haze = haze * (1.0 - 0.85 * freewater).clamp(0.0, 1.0)

        return {
            "freewater": freewater.detach().clamp(0.0, 1.0),
            "haze": haze.detach().clamp(0.0, 1.0),
            "support": spatial_support.detach().clamp(0.0, 1.0),
            "anchor_similarity": anchor_similarity.detach().clamp(0.0, 1.0),
            "input_chroma": input_chroma.detach().clamp(0.0, 1.0),
            "scene_protection": scene_protection.detach().clamp(0.0, 1.0),
            "low_structure": low_structure.detach().clamp(0.0, 1.0),
        }

    def _build_layered_veil_scene_factorization(
        self,
        depth_confidence: torch.Tensor,
        transition_depth_confidence: torch.Tensor,
        texture_confidence: torch.Tensor,
        intrinsic_texture_confidence: torch.Tensor,
        low_depth_edge_confidence: torch.Tensor,
        depth_edge_support: torch.Tensor,
        scene_rescue: torch.Tensor,
        residual_water_probability: torch.Tensor,
        pure_probability: torch.Tensor,
        transition_probability: torch.Tensor,
        connected_freewater: torch.Tensor,
        anchored_matte: Dict[str, torch.Tensor],
        gt_underwater_image: torch.Tensor,
        water_background_image: torch.Tensor,
        intrinsic_image: Optional[torch.Tensor],
        accumulation: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Cross-view-ready layer factorization for water-veiled scene content.

        Empty freewater is allowed to become black. Attached veil is explicitly
        separated from empty water: it is water-colored material in front of an
        object carrier and should be de-veiled, not erased. Scene core is a
        stricter replacement for raw-intrinsic rescue; water-colored context can
        veto scene protection even if raw intrinsic has weak edges or alpha.
        """
        reference = depth_confidence[..., :1]
        zero = torch.zeros_like(reference)
        if not self.config.enable_layered_veil_scene_factorization:
            uncertainty = (1.0 - torch.maximum(anchored_matte["freewater"], scene_rescue)).clamp(0.0, 1.0)
            return {
                "empty_freewater": anchored_matte["freewater"].detach().clamp(0.0, 1.0),
                "attached_veil": anchored_matte["haze"].detach().clamp(0.0, 1.0),
                "scene_core": scene_rescue.detach().clamp(0.0, 1.0),
                "uncertain_boundary": uncertainty.detach(),
                "water_color_context": anchored_matte["anchor_similarity"].detach().clamp(0.0, 1.0),
                "veil_alpha": zero,
            }

        rgb = torch.nan_to_num(
            gt_underwater_image[..., :3].detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        water_bg = torch.nan_to_num(
            water_background_image[..., :3].detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        red = rgb[..., 0:1]
        green = rgb[..., 1:2]
        blue = rgb[..., 2:3]
        chroma_scale = max(float(self.config.water_anchor_chroma_scale), 1e-4)
        input_water_chroma = anchored_matte["input_chroma"].detach().clamp(0.0, 1.0)
        anchor_similarity = anchored_matte["anchor_similarity"].detach().clamp(0.0, 1.0)
        low_structure = anchored_matte["low_structure"].detach().clamp(0.0, 1.0)
        anchored_freewater = anchored_matte["freewater"].detach().clamp(0.0, 1.0)
        anchored_haze = anchored_matte["haze"].detach().clamp(0.0, 1.0)

        water_bg_distance = torch.abs(rgb - water_bg).mean(dim=-1, keepdim=True)
        water_bg_similarity = torch.exp(
            -water_bg_distance / max(float(self.config.water_anchor_color_sigma) * 1.25, 1e-4)
        ).clamp(0.0, 1.0)
        intrinsic_blue_water = zero
        raw_nonwater_color = zero
        raw_visible_luma = zero
        raw_texture_scene = (1.0 - intrinsic_texture_confidence.detach().clamp(0.0, 1.0)).clamp(0.0, 1.0)
        if intrinsic_image is not None and intrinsic_image.numel() > 0:
            intrinsic = torch.nan_to_num(
                intrinsic_image[..., :3].detach(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            intrinsic_blue_water = self._intrinsic_blue_water_chroma(intrinsic)
            intrinsic_luma = intrinsic.mean(dim=-1, keepdim=True)
            luma_threshold = max(self.config.intrinsic_rescue_luma_threshold, 1e-4)
            raw_visible_luma = torch.sigmoid((intrinsic_luma - luma_threshold) / luma_threshold * 4.0)
            red_green = torch.maximum(intrinsic[..., 0:1], intrinsic[..., 1:2])
            raw_nonwater_color = torch.sigmoid(
                (red_green - 0.92 * intrinsic[..., 2:3]) / chroma_scale * 3.0
            ) * raw_visible_luma

        # Water context is deliberately broad: attached veil is often textured
        # and edge-like, so it cannot require low structure.  It combines input
        # water chroma, water-color anchor similarity, WPP ambient similarity,
        # and raw-intrinsic blue/cyan residuals.
        water_color_context = torch.maximum(
            torch.maximum(anchor_similarity * input_water_chroma, water_bg_similarity * input_water_chroma),
            torch.maximum(residual_water_probability.detach().clamp(0.0, 1.0), intrinsic_blue_water),
        ).clamp(0.0, 1.0)
        water_color_context = torch.maximum(
            water_color_context,
            transition_probability.detach().clamp(0.0, 1.0) * input_water_chroma,
        ).clamp(0.0, 1.0)

        texture_scene = (1.0 - texture_confidence.detach().clamp(0.0, 1.0)).clamp(0.0, 1.0)
        structure_context = torch.maximum(
            depth_edge_support.detach().clamp(0.0, 1.0),
            torch.maximum(texture_scene, raw_texture_scene),
        ).clamp(0.0, 1.0)
        del accumulation

        nonwater_input = torch.sigmoid(
            (torch.maximum(red, green) - 0.94 * blue) / chroma_scale * 3.0
        ) * (1.0 - 0.70 * input_water_chroma).clamp(0.0, 1.0)
        water_veto = (
            self.config.layer_scene_core_water_suppression
            * water_color_context
            * transition_depth_confidence.detach().clamp(0.0, 1.0)
        ).clamp(0.0, 0.95)
        scene_core_from_raw = raw_nonwater_color * (0.35 + 0.65 * structure_context)
        scene_core_from_input = nonwater_input * (0.30 + 0.70 * structure_context)
        scene_core = torch.maximum(scene_core_from_raw, scene_core_from_input)
        scene_core = torch.maximum(scene_core, scene_rescue.detach().clamp(0.0, 1.0) * (1.0 - water_veto))
        scene_water_cancel = (
            water_color_context
            * transition_depth_confidence.detach().clamp(0.0, 1.0)
            * (1.0 - raw_nonwater_color).clamp(0.0, 1.0)
            * (1.0 - nonwater_input).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        scene_core = scene_core * (1.0 - 0.72 * scene_water_cancel).clamp(0.08, 1.0)
        scene_core = scene_core * (1.0 - 0.55 * water_color_context * low_structure).clamp(0.10, 1.0)
        scene_core = scene_core.clamp(0.0, 1.0)

        carrier_context = structure_context
        no_carrier_context = (1.0 - carrier_context).clamp(0.0, 1.0)
        empty_seed = torch.maximum(
            torch.maximum(pure_probability.detach().clamp(0.0, 1.0), connected_freewater.detach().clamp(0.0, 1.0)),
            anchored_freewater,
        )
        empty_freewater = torch.maximum(
            empty_seed,
            water_color_context
            * transition_depth_confidence.detach().clamp(0.0, 1.0)
            * low_structure
            * (0.35 + 0.65 * no_carrier_context),
        )
        empty_freewater = empty_freewater * (1.0 - 0.45 * carrier_context * water_color_context).clamp(0.25, 1.0)
        empty_freewater = empty_freewater * (1.0 - 0.88 * scene_core).clamp(0.02, 1.0)
        empty_freewater = (empty_freewater * self.config.layer_empty_freewater_strength).clamp(0.0, 1.0)

        attached_veil = torch.maximum(
            anchored_haze,
            water_color_context
            * transition_depth_confidence.detach().clamp(0.0, 1.0)
            * carrier_context
            * (0.25 + 0.75 * structure_context),
        )
        attached_veil = attached_veil * (1.0 - 0.70 * empty_freewater).clamp(0.0, 1.0)
        attached_veil = attached_veil * (1.0 - 0.30 * scene_core).clamp(0.08, 1.0)
        attached_veil = (attached_veil * self.config.layer_attached_veil_strength).clamp(0.0, 1.0)

        layer_sum = empty_freewater + attached_veil + scene_core
        normalizer = torch.maximum(layer_sum, torch.ones_like(layer_sum))
        empty_freewater = empty_freewater / normalizer
        attached_veil = attached_veil / normalizer
        scene_core = scene_core / normalizer
        uncertain_boundary = (1.0 - torch.maximum(torch.maximum(empty_freewater, attached_veil), scene_core)).clamp(0.0, 1.0)
        veil_alpha = (
            attached_veil
            + self.config.layer_uncertain_deveil_strength * uncertain_boundary * water_color_context
        ) * max(min(float(self.config.layer_attached_veil_max_alpha), 0.95), 0.0)
        veil_alpha = veil_alpha * (1.0 - 0.45 * scene_core).clamp(0.0, 1.0)
        veil_alpha = veil_alpha.clamp(0.0, max(min(float(self.config.layer_attached_veil_max_alpha), 0.95), 0.0))
        return {
            "empty_freewater": empty_freewater.detach().clamp(0.0, 1.0),
            "attached_veil": attached_veil.detach().clamp(0.0, 1.0),
            "scene_core": scene_core.detach().clamp(0.0, 1.0),
            "uncertain_boundary": uncertain_boundary.detach().clamp(0.0, 1.0),
            "water_color_context": water_color_context.detach().clamp(0.0, 1.0),
            "veil_alpha": veil_alpha.detach().clamp(0.0, 1.0),
        }

    def _compose_layered_clean_render(
        self,
        intrinsic_raw: torch.Tensor,
        water_background_image: torch.Tensor,
        layered: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compose clean output from empty freewater, attached veil, scene core and uncertainty."""
        raw = torch.nan_to_num(intrinsic_raw[..., :3], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        water_bg = torch.nan_to_num(
            water_background_image[..., :3].detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        veil_alpha = layered.get("veil_alpha", torch.zeros_like(raw[..., :1])).to(raw).clamp(0.0, 0.95)
        empty_freewater = layered.get("empty_freewater", torch.zeros_like(raw[..., :1])).to(raw).clamp(0.0, 1.0)
        attached_veil = layered.get("attached_veil", torch.zeros_like(raw[..., :1])).to(raw).clamp(0.0, 1.0)
        scene_core = layered.get("scene_core", torch.zeros_like(raw[..., :1])).to(raw).clamp(0.0, 1.0)
        uncertain = layered.get("uncertain_boundary", torch.zeros_like(raw[..., :1])).to(raw).clamp(0.0, 1.0)
        water_context = layered.get("water_color_context", torch.zeros_like(raw[..., :1])).to(raw).clamp(0.0, 1.0)

        de_veiled = (raw - veil_alpha * water_bg) / (1.0 - veil_alpha).clamp_min(1e-4)
        de_veiled = de_veiled.clamp(0.0, 1.0)
        min_retained = raw * (1.0 - max(min(float(self.config.layer_deveil_darkening_limit), 1.0), 0.0))
        de_veiled = torch.maximum(de_veiled, min_retained)

        veil_blend = (attached_veil + self.config.layer_uncertain_deveil_strength * uncertain * water_context).clamp(0.0, 1.0)
        clean = torch.lerp(raw, de_veiled, veil_blend)
        scene_keep = (scene_core * (1.0 - attached_veil)).clamp(0.0, 1.0)
        clean = torch.lerp(clean, raw, 0.20 * scene_keep)
        clean = clean * (1.0 - empty_freewater).clamp(0.0, 1.0)
        hard_empty = (
            (empty_freewater >= self.config.layer_empty_freewater_strict_threshold)
            & (scene_core <= 0.20)
        )
        clean = torch.where(hard_empty, torch.zeros_like(clean), clean)
        return {
            "clean": clean.clamp(0.0, 1.0),
            "deveiled": de_veiled.clamp(0.0, 1.0),
            "veil_blend": veil_blend.clamp(0.0, 1.0),
        }

    def _build_water_scene_responsibility_field(
        self,
        pure_probability: torch.Tensor,
        transition_probability: torch.Tensor,
        residual_water_probability: torch.Tensor,
        connected_freewater: torch.Tensor,
        scene_rescue: torch.Tensor,
        depth_edge_support: torch.Tensor,
        color_residual_support: torch.Tensor,
        intrinsic_image: Optional[torch.Tensor],
        accumulation: Optional[torch.Tensor],
        intrinsic_texture: Optional[torch.Tensor],
        anchored_freewater: Optional[torch.Tensor] = None,
        anchored_haze: Optional[torch.Tensor] = None,
        anchored_scene_protection: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Softly assign every pixel to freewater, residual haze, scene, or uncertainty.

        This replaces the removed Scene-Safe Dual Topology Matte. The key
        difference is that uncertain / weakly visible regions are not converted
        into hard water masks. Destructive losses only see high-confidence
        freewater; residual haze remains a non-destructive color-removal cue.
        """
        reference = pure_probability[..., :1]
        zero = torch.zeros_like(reference)
        if not self.config.enable_water_scene_responsibility_field:
            scene = scene_rescue.detach().clamp(0.0, 1.0)
            freewater = torch.maximum(pure_probability, connected_freewater).detach().clamp(0.0, 1.0)
            haze = residual_water_probability.detach().clamp(0.0, 1.0)
            uncertainty = (1.0 - torch.maximum(torch.maximum(freewater, haze), scene)).clamp(0.0, 1.0)
            return {
                "freewater": freewater,
                "haze": haze,
                "scene": scene,
                "uncertainty": uncertainty,
            }

        if intrinsic_image is not None and intrinsic_image.numel() > 0:
            intrinsic = torch.nan_to_num(
                intrinsic_image[..., :3].detach(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            luma = intrinsic.mean(dim=-1, keepdim=True)
            saturation = intrinsic.amax(dim=-1, keepdim=True) - intrinsic.amin(dim=-1, keepdim=True)
            red_or_green = torch.maximum(intrinsic[..., 0:1], intrinsic[..., 1:2])
            blue = intrinsic[..., 2:3]
            non_blue_scene = torch.sigmoid((red_or_green - 0.88 * blue) / 0.05)
            visible_luma = torch.sigmoid(
                (luma - self.config.intrinsic_rescue_luma_threshold)
                / max(self.config.intrinsic_rescue_luma_threshold, 1e-4)
                * 5.0
            )
            saturation_scene = torch.sigmoid((saturation - 0.08) / 0.08 * 5.0)
            if intrinsic_texture is None:
                intrinsic_texture = self._image_gradient_magnitude(intrinsic).detach()
            texture = torch.nan_to_num(intrinsic_texture.detach(), nan=0.0, posinf=1.0, neginf=0.0)
            texture_cut = torch.quantile(
                texture.flatten(),
                min(max(self.config.intrinsic_rescue_texture_quantile, 0.0), 1.0),
            ).detach().clamp_min(1e-5)
            texture_scene = torch.sigmoid(
                (texture - texture_cut)
                / texture_cut
                * self.config.water_null_texture_sharpness
            )
        else:
            visible_luma = zero
            non_blue_scene = zero
            saturation_scene = zero
            texture_scene = zero

        del accumulation

        scene_structure = torch.maximum(
            torch.maximum(depth_edge_support.detach().clamp(0.0, 1.0), texture_scene),
            torch.maximum(non_blue_scene, saturation_scene),
        )
        scene_evidence = torch.maximum(
            scene_rescue.detach().clamp(0.0, 1.0) * self.config.responsibility_scene_strength,
            visible_luma * scene_structure,
        )
        if anchored_scene_protection is not None:
            scene_evidence = torch.maximum(
                scene_evidence,
                anchored_scene_protection.detach().clamp(0.0, 1.0) * 0.85,
            )
        scene_evidence = scene_evidence.clamp(0.0, 1.0)

        freewater_seed = torch.maximum(
            pure_probability.detach().clamp(0.0, 1.0),
            connected_freewater.detach().clamp(0.0, 1.0) * self.config.connected_freewater_strength,
        )
        if anchored_freewater is not None:
            freewater_seed = torch.maximum(
                freewater_seed,
                anchored_freewater.detach().clamp(0.0, 1.0) * self.config.water_anchor_matte_strength,
            )
        freewater_guard = (
            1.0
            - 0.85 * scene_evidence
            - 0.55 * depth_edge_support.detach().clamp(0.0, 1.0)
            - 0.35 * visible_luma * non_blue_scene
        ).clamp(0.02, 1.0)
        freewater_evidence = (freewater_seed * freewater_guard * self.config.responsibility_freewater_strength).clamp(0.0, 1.0)

        haze_seed = torch.maximum(
            residual_water_probability.detach().clamp(0.0, 1.0),
            transition_probability.detach().clamp(0.0, 1.0) * 0.55,
        )
        if anchored_haze is not None:
            haze_seed = torch.maximum(haze_seed, anchored_haze.detach().clamp(0.0, 1.0))
        haze_guard = (1.0 - 0.55 * freewater_evidence).clamp(0.05, 1.0)
        haze_scene_support = (0.35 + 0.65 * scene_evidence).clamp(0.0, 1.0)
        haze_evidence = (
            haze_seed
            * haze_guard
            * haze_scene_support
            * self.config.responsibility_haze_strength
        ).clamp(0.0, 1.0)

        # Direct evidence split. The earlier softmax made scene evidence suppress
        # haze/freewater almost everywhere, which caused water branches to have
        # little training effect. Here freewater, haze and scene can co-exist;
        # only over-confident sums are softly normalized back to a valid budget.
        freewater = freewater_evidence.clamp(0.0, 1.0)
        haze = (haze_evidence * (1.0 - 0.80 * freewater)).clamp(0.0, 1.0)
        scene = torch.maximum(
            scene_evidence * (1.0 - 0.55 * freewater).clamp(0.0, 1.0),
            scene_rescue.detach().clamp(0.0, 1.0) * 0.75,
        ).clamp(0.0, 1.0)
        total = freewater + haze + scene
        normalizer = torch.maximum(total, torch.ones_like(total))
        freewater = freewater / normalizer
        haze = haze / normalizer
        scene = scene / normalizer
        uncertainty = (1.0 - total.clamp(0.0, 1.0)).clamp(0.0, 1.0)
        return {
            "freewater": freewater.detach(),
            "haze": haze.detach(),
            "scene": scene.detach(),
            "uncertainty": uncertainty.detach(),
        }

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

        del accumulation

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
        horizon_active = (
            top_strength.mean() >= self.config.water_horizon_min_confidence
        ).to(score)

        y = torch.linspace(0.0, 1.0, height, device=score.device, dtype=score.dtype)
        top_bias = (1.0 - y).clamp_min(0.0)[:, None]
        weights = score * top_bias
        boundary = (weights * y[:, None]).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-6)
        boundary = (boundary + self.config.water_horizon_band).clamp(0.0, 1.0)

        band = max(self.config.water_horizon_band, 1e-3)
        prior = torch.sigmoid((boundary[None, :] - y[:, None]) / band * 6.0)
        prior = (
            prior
            * top_strength[None, :].clamp(0.0, 1.0)
            * self.config.water_horizon_strength
            * horizon_active
        )
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
                "anchored_freewater": zero,
                "anchored_haze": zero,
                "water_anchor_similarity": zero,
                "water_anchor_input_chroma": zero,
                "water_matte_scene_protection": zero,
                "water_matte_low_structure": zero,
                "empty_freewater": zero,
                "attached_veil": zero,
                "scene_core": zero,
                "uncertain_boundary": torch.ones_like(zero),
                "water_color_context": zero,
                "veil_alpha": zero,
                "responsibility_freewater": zero,
                "responsibility_haze": zero,
                "responsibility_scene": zero,
                "responsibility_uncertainty": torch.ones_like(zero),
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

        anchored_matte = self._build_physics_anchored_water_matte(
            depth_confidence,
            transition_depth_confidence,
            texture_confidence,
            intrinsic_texture_confidence,
            low_depth_edge_confidence,
            edge_protection,
            scene_rescue,
            depth_edge_support,
            residual_water_probability,
            gt_underwater_image,
            water_background_image,
            intrinsic_image,
            accumulation,
        )

        connected_seed = torch.maximum(
            torch.maximum(pure_base.detach(), residual_water_probability),
            anchored_matte["freewater"] * 0.75,
        )
        connected_support = torch.maximum(
            torch.maximum(transition_base.detach(), residual_water_probability),
            anchored_matte["support"],
        )
        connected_support = connected_support * (
            1.0 - 0.50 * scene_rescue
        ).clamp(0.15, 1.0)
        connected_freewater = self._build_connected_freewater_prior(
            connected_seed.clamp(0.0, 1.0),
            connected_support.clamp(0.0, 1.0),
        )

        layered = self._build_layered_veil_scene_factorization(
            depth_confidence,
            transition_depth_confidence,
            texture_confidence,
            intrinsic_texture_confidence,
            low_depth_edge_confidence,
            depth_edge_support,
            scene_rescue,
            residual_water_probability,
            pure_probability,
            transition_probability,
            connected_freewater,
            anchored_matte,
            gt_underwater_image,
            water_background_image,
            intrinsic_image,
            accumulation,
        )

        responsibility = self._build_water_scene_responsibility_field(
            pure_probability,
            transition_probability,
            residual_water_probability,
            connected_freewater,
            scene_rescue,
            depth_edge_support,
            color_residual_support,
            intrinsic_image,
            accumulation,
            intrinsic_texture,
            torch.maximum(anchored_matte["freewater"], layered["empty_freewater"]),
            torch.maximum(anchored_matte["haze"], layered["attached_veil"]),
            layered["scene_core"],
        )
        freewater_responsibility = responsibility["freewater"]
        haze_responsibility = responsibility["haze"]

        null_probability = torch.maximum(
            torch.maximum(
                freewater_responsibility * self.config.responsibility_freewater_strength,
                layered["empty_freewater"],
            ),
            haze_responsibility * self.config.responsibility_haze_strength,
        )
        null_probability = torch.maximum(null_probability, layered["attached_veil"] * 0.45)
        null_probability = torch.maximum(null_probability, pure_probability * 0.65)

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
            "anchored_freewater": anchored_matte["freewater"],
            "anchored_haze": anchored_matte["haze"],
            "water_anchor_similarity": anchored_matte["anchor_similarity"],
            "water_anchor_input_chroma": anchored_matte["input_chroma"],
            "water_matte_scene_protection": anchored_matte["scene_protection"],
            "water_matte_low_structure": anchored_matte["low_structure"],
            "empty_freewater": layered["empty_freewater"],
            "attached_veil": layered["attached_veil"],
            "scene_core": layered["scene_core"],
            "uncertain_boundary": layered["uncertain_boundary"],
            "water_color_context": layered["water_color_context"],
            "veil_alpha": layered["veil_alpha"],
            "responsibility_freewater": responsibility["freewater"],
            "responsibility_haze": responsibility["haze"],
            "responsibility_scene": responsibility["scene"],
            "responsibility_uncertainty": responsibility["uncertainty"],
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU-only foreground support from pseudo-depth morphology."""
        downscale = self._get_downscale_factor()
        if image_idx in self.foreground_mask_cache.keys() and downscale in self.foreground_mask_cache[image_idx].keys():
            cached = self.foreground_mask_cache[image_idx][downscale]
            return cached["mask_tensor"], cached["background_pixel_ratio"]

        background_depth_threshold = 1e-2
        foreground_chw = (pseudo_depth[..., :1].detach() > background_depth_threshold).to(
            pseudo_depth.dtype
        ).permute(2, 0, 1).unsqueeze(0)
        # Closing fills small depth holes without a CPU contour pass. A following
        # opening removes isolated pseudo-depth responses in empty water.
        foreground_chw = F.max_pool2d(foreground_chw, 5, stride=1, padding=2)
        foreground_chw = -F.max_pool2d(-foreground_chw, 5, stride=1, padding=2)
        foreground_chw = -F.max_pool2d(-foreground_chw, 3, stride=1, padding=1)
        foreground_chw = F.max_pool2d(foreground_chw, 3, stride=1, padding=1)
        foreground_mask = foreground_chw.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)
        background_pixel_ratio = (1.0 - foreground_mask).mean().detach()

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
        if (
            "empty_freewater_exposure_score" not in self.strategy_state
            or self.strategy_state["empty_freewater_exposure_score"].shape[0] != n
            or self.strategy_state["empty_freewater_exposure_score"].device != device
        ):
            self.strategy_state["empty_freewater_exposure_score"] = torch.zeros(n, device=device)
        if (
            "attached_veil_exposure_score" not in self.strategy_state
            or self.strategy_state["attached_veil_exposure_score"].shape[0] != n
            or self.strategy_state["attached_veil_exposure_score"].device != device
        ):
            self.strategy_state["attached_veil_exposure_score"] = torch.zeros(n, device=device)
        if (
            "scene_core_exposure_score" not in self.strategy_state
            or self.strategy_state["scene_core_exposure_score"].shape[0] != n
            or self.strategy_state["scene_core_exposure_score"].device != device
        ):
            self.strategy_state["scene_core_exposure_score"] = torch.zeros(n, device=device)
        self.water_exposure_score = self.strategy_state["water_exposure_score"]
        self.scene_exposure_score = self.strategy_state["scene_exposure_score"]
        self.depth_edge_support_score = self.strategy_state["depth_edge_support_score"]
        self.color_residual_support_score = self.strategy_state["color_residual_support_score"]
        self.view_consistency_score = self.strategy_state["view_consistency_score"]
        self.empty_freewater_exposure_score = self.strategy_state["empty_freewater_exposure_score"]
        self.attached_veil_exposure_score = self.strategy_state["attached_veil_exposure_score"]
        self.scene_core_exposure_score = self.strategy_state["scene_core_exposure_score"]

    @staticmethod
    def _sample_projection_image(
        image: torch.Tensor,
        means2d: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinearly sample an HWC image at projected Gaussian centres on-device."""
        if image.numel() == 0 or means2d.numel() == 0:
            channels = image.shape[-1] if image.dim() == 3 else 1
            return means2d.new_zeros((means2d.shape[0], channels))
        height, width = image.shape[:2]
        x = 2.0 * means2d[:, 0] / max(width - 1, 1) - 1.0
        y = 2.0 * means2d[:, 1] / max(height - 1, 1) - 1.0
        grid = torch.stack([x, y], dim=-1).reshape(1, 1, -1, 2)
        sampled = F.grid_sample(
            image.permute(2, 0, 1).unsqueeze(0).float(),
            grid.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled[0, :, 0].transpose(0, 1).to(means2d)

    @torch.no_grad()
    def _gaussian_geometric_counterfactual_support(
        self,
        pseudo_depth: Optional[torch.Tensor],
        rendered_depth: Optional[torch.Tensor],
        depth_edge_support: torch.Tensor,
        low_structure_probability: torch.Tensor,
        water_only_advantage: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Infer per-Gaussian geometry support without another rasterization."""
        if pseudo_depth is None or not hasattr(self, "info") or self.info is None:
            return None, None
        means2d = self.info.get("means2d")
        projected_depths = self.info.get("depths")
        radii = self.info.get("radii")
        if (
            means2d is None
            or projected_depths is None
            or radii is None
            or means2d.dim() != 3
            or means2d.shape[0] != 1
        ):
            return None, None
        xy = means2d[0].detach()
        z = projected_depths[0].detach().reshape(-1)
        visible = radii[0].detach().reshape(-1) > 0
        height, width = pseudo_depth.shape[:2]
        visible = (
            visible
            & torch.isfinite(xy).all(dim=-1)
            & torch.isfinite(z)
            & (z > 1e-5)
            & (xy[:, 0] >= 0.0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0.0)
            & (xy[:, 1] < height)
        )

        pseudo = self._sample_projection_image(pseudo_depth[..., :1].detach(), xy)[:, 0]
        edge = self._sample_projection_image(depth_edge_support[..., :1].detach(), xy)[:, 0]
        low_structure = self._sample_projection_image(
            low_structure_probability[..., :1].detach(), xy
        )[:, 0]
        counterfactual = self._sample_projection_image(
            water_only_advantage[..., :1].detach(), xy
        )[:, 0]
        threshold = max(float(self.config.water_null_depth_threshold), 1e-4)
        pseudo_surface = torch.sigmoid((pseudo - threshold) / threshold * 4.0)

        if rendered_depth is not None and rendered_depth.numel() > 0:
            expected_z = self._sample_projection_image(
                rendered_depth[..., :1].detach(), xy
            )[:, 0].clamp_min(1e-4)
            relative_error = torch.abs(z - expected_z) / (0.10 * expected_z + 0.02)
            front_consistency = torch.exp(-relative_error).clamp(0.0, 1.0)
        else:
            front_consistency = torch.ones_like(pseudo_surface)

        strong_surface_anchor = (
            pseudo_surface
            * front_consistency
            * edge.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        surface_support = pseudo_surface * (
            0.30 + 0.70 * front_consistency
        ) * (0.40 + 0.60 * edge.clamp(0.0, 1.0)) * (
            1.0 - 0.80 * counterfactual.clamp(0.0, 1.0)
        ).clamp(0.05, 1.0)
        surface_support = torch.maximum(
            surface_support,
            (1.0 - counterfactual.clamp(0.0, 1.0))
            * pseudo_surface
            * (0.35 + 0.65 * front_consistency),
        )
        surface_support = torch.maximum(surface_support, 0.90 * strong_surface_anchor)
        medium_support = (
            (1.0 - pseudo_surface)
            * (0.30 + 0.70 * low_structure.clamp(0.0, 1.0))
            * (0.40 + 0.60 * counterfactual.clamp(0.0, 1.0))
        )
        medium_support = torch.maximum(
            medium_support,
            counterfactual.clamp(0.0, 1.0)
            * low_structure.clamp(0.0, 1.0)
            * (1.0 - 0.95 * strong_surface_anchor).clamp(0.05, 1.0),
        )
        validity = visible.to(surface_support)
        return (
            (medium_support * validity).clamp(0.0, 1.0),
            (surface_support * validity).clamp(0.0, 1.0),
        )

    @torch.no_grad()
    def _update_water_lifecycle_scores(
        self,
        water_probability: torch.Tensor,
        scene_probability: Optional[torch.Tensor] = None,
        haze_probability: Optional[torch.Tensor] = None,
        depth_edge_support: Optional[torch.Tensor] = None,
        color_residual_support: Optional[torch.Tensor] = None,
        empty_freewater_probability: Optional[torch.Tensor] = None,
        attached_veil_probability: Optional[torch.Tensor] = None,
        scene_core_probability: Optional[torch.Tensor] = None,
        water_similarity_probability: Optional[torch.Tensor] = None,
        low_structure_probability: Optional[torch.Tensor] = None,
        water_only_advantage: Optional[torch.Tensor] = None,
        pseudo_depth: Optional[torch.Tensor] = None,
        rendered_depth: Optional[torch.Tensor] = None,
        view_id: Optional[torch.Tensor] = None,
        valid_image_mask: Optional[torch.Tensor] = None,
    ):
        """Accumulate per-Gaussian cross-view responsibility evidence."""
        if not self.training or not self.config.enable_water_null_occupancy:
            return
        if not hasattr(self, "info") or self.info is None:
            return
        means2d = self.info.get("means2d")
        radii = self.info.get("radii")
        opacities = self.info.get("opacities")
        if means2d is None or radii is None or means2d.dim() != 3:
            return

        height, width = water_probability.shape[:2]
        if scene_probability is None:
            scene_probability = (1.0 - water_probability).clamp(0.0, 1.0)
        if haze_probability is None:
            haze_probability = torch.zeros_like(water_probability)
        if depth_edge_support is None:
            depth_edge_support = torch.zeros_like(water_probability)
        if color_residual_support is None:
            color_residual_support = haze_probability
        if empty_freewater_probability is None:
            empty_freewater_probability = water_probability
        if attached_veil_probability is None:
            attached_veil_probability = haze_probability
        if scene_core_probability is None:
            scene_core_probability = scene_probability
        if water_similarity_probability is None:
            water_similarity_probability = water_probability
        if low_structure_probability is None:
            low_structure_probability = (1.0 - depth_edge_support).clamp(0.0, 1.0)
        if water_only_advantage is None:
            water_only_advantage = torch.zeros_like(water_probability)
        if (
            self._uses_surface_carrier()
            and view_id is not None
            and self.step % self.config.carrier_update_every == 0
        ):
            water_context = (
                water_similarity_probability.detach().clamp(0.0, 1.0)
                * low_structure_probability.detach().clamp(0.0, 1.0)
            )
            medium_probability = torch.maximum(
                empty_freewater_probability.detach().clamp(0.0, 1.0),
                attached_veil_probability.detach().clamp(0.0, 1.0) * water_context,
            )
            medium_probability = torch.maximum(
                medium_probability,
                water_only_advantage.detach().clamp(0.0, 1.0) * water_context,
            )
            # Scene-core colour is only geometric evidence when it is backed by
            # structure or disagrees with the physical water-only hypothesis.
            scene_support = (1.0 - water_context).clamp(0.0, 1.0)
            surface_probability = torch.maximum(
                depth_edge_support.detach().clamp(0.0, 1.0),
                scene_core_probability.detach().clamp(0.0, 1.0)
                * (0.15 + 0.85 * scene_support),
            )
            ambiguity = torch.minimum(medium_probability, surface_probability)
            uncertainty_probability = torch.maximum(
                (1.0 - torch.maximum(medium_probability, surface_probability)).clamp(0.0, 1.0),
                0.5 * ambiguity,
            )
            responsibility_mass = self.surface_carrier.rasterize_responsibility_mass(
                info=self.info,
                medium=medium_probability,
                surface=surface_probability,
                uncertainty=uncertainty_probability,
                valid_image_mask=valid_image_mask,
            )
            geometric_medium, geometric_surface = (
                self._gaussian_geometric_counterfactual_support(
                    pseudo_depth,
                    rendered_depth,
                    depth_edge_support,
                    low_structure_probability,
                    water_only_advantage,
                )
            )
            self.surface_carrier.sync_from_strategy_state(self.strategy_state)
            self.surface_carrier.update_from_responsibilities(
                step=self.step,
                view_id=view_id,
                info=self.info,
                medium=medium_probability,
                surface=surface_probability,
                uncertainty=uncertainty_probability,
                responsibility_mass=responsibility_mass,
                valid_image_mask=valid_image_mask,
                means=self.means,
                camera_origin=getattr(self, "_last_carrier_camera_origin", None),
                geometric_medium_support=geometric_medium,
                geometric_surface_support=geometric_surface,
            )
            support_interval = max(int(self.config.water_support_update_every), 1)
            if self.step % support_interval == 0:
                carrier = self.surface_carrier.render_weights()
                support_confidence = (
                    0.15 + 0.85 * (1.0 - carrier["uncertainty"])
                ).clamp(0.0, 1.0)
                self.path_integrated_renderer.field.update_free_space_support(
                    self.means,
                    carrier["medium_fraction"],
                    carrier["surface_fraction"],
                    confidence=support_confidence,
                    opacity=torch.sigmoid(self.opacities).squeeze(-1),
                    camera_origin=getattr(self, "_last_carrier_camera_origin", None),
                )

        if not self._uses_legacy_water_lifecycle():
            return
        self._ensure_water_lifecycle_buffers()
        water_score = self.strategy_state["water_exposure_score"]
        scene_score = self.strategy_state["scene_exposure_score"]
        depth_edge_score = self.strategy_state["depth_edge_support_score"]
        color_residual_score = self.strategy_state["color_residual_support_score"]
        view_consistency_score = self.strategy_state["view_consistency_score"]
        empty_score = self.strategy_state["empty_freewater_exposure_score"]
        attached_score = self.strategy_state["attached_veil_exposure_score"]
        scene_core_score = self.strategy_state["scene_core_exposure_score"]
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
        scene_sum = torch.zeros_like(x_float, device=scene_score.device)
        haze_sum = torch.zeros_like(x_float, device=color_residual_score.device)
        edge_sum = torch.zeros_like(x_float, device=water_score.device)
        residual_sum = torch.zeros_like(x_float, device=water_score.device)
        empty_sum = torch.zeros_like(x_float, device=water_score.device)
        attached_sum = torch.zeros_like(x_float, device=water_score.device)
        scene_core_sum = torch.zeros_like(x_float, device=water_score.device)
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
            sampled_scene = scene_probability[y, x, 0].to(scene_score.device)
            sampled_haze = haze_probability[y, x, 0].to(color_residual_score.device)
            sampled_edge = depth_edge_support[y, x, 0].to(water_score.device)
            sampled_residual = color_residual_support[y, x, 0].to(water_score.device)
            sampled_empty = empty_freewater_probability[y, x, 0].to(water_score.device)
            sampled_attached = attached_veil_probability[y, x, 0].to(water_score.device)
            sampled_scene_core = scene_core_probability[y, x, 0].to(water_score.device)
            sample_valid_float = sample_valid.to(water_score.device)
            water_sum += sampled_water * sample_valid_float
            scene_sum += sampled_scene * sample_valid_float
            haze_sum += sampled_haze * sample_valid_float
            edge_sum += sampled_edge * sample_valid_float
            residual_sum += sampled_residual * sample_valid_float
            empty_sum += sampled_empty * sample_valid_float
            attached_sum += sampled_attached * sample_valid_float
            scene_core_sum += sampled_scene_core * sample_valid_float
            sample_count += sample_valid_float
        water_at_mean = water_sum / sample_count.clamp_min(1.0)
        scene_at_mean = scene_sum / sample_count.clamp_min(1.0)
        haze_at_mean = haze_sum / sample_count.clamp_min(1.0)
        edge_at_mean = edge_sum / sample_count.clamp_min(1.0)
        residual_at_mean = residual_sum / sample_count.clamp_min(1.0)
        empty_at_mean = empty_sum / sample_count.clamp_min(1.0)
        attached_at_mean = attached_sum / sample_count.clamp_min(1.0)
        scene_core_at_mean = scene_core_sum / sample_count.clamp_min(1.0)
        if opacities is not None and opacities.dim() >= 2:
            visibility_weight = opacities[0].detach().flatten().to(water_at_mean.device)
        else:
            visibility_weight = torch.sigmoid(self.opacities.detach().flatten()).to(water_at_mean.device)
        visibility_weight = torch.where(in_bounds.to(water_at_mean.device), visibility_weight, torch.zeros_like(visibility_weight))

        ema = min(max(self.config.water_lifecycle_ema, 0.0), 0.999)
        update = 1.0 - ema
        observed = in_bounds.to(water_score.device)
        water_update = (water_at_mean * visibility_weight).clamp(0.0, 1.0)
        scene_update = (scene_at_mean * visibility_weight).clamp(0.0, 1.0)
        edge_update = (edge_at_mean * visibility_weight).clamp(0.0, 1.0)
        residual_update = (torch.maximum(haze_at_mean, residual_at_mean) * visibility_weight).clamp(0.0, 1.0)
        empty_update = (empty_at_mean * visibility_weight).clamp(0.0, 1.0)
        attached_update = (attached_at_mean * visibility_weight).clamp(0.0, 1.0)
        scene_core_update = (scene_core_at_mean * visibility_weight).clamp(0.0, 1.0)
        max_responsibility = torch.maximum(torch.maximum(water_at_mean, scene_at_mean), haze_at_mean)
        consistency_update = ((max_responsibility - 1.0 / 3.0).clamp_min(0.0) * 1.5 * visibility_weight).clamp(0.0, 1.0)
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
        empty_score = torch.where(
            observed,
            ema * empty_score + update * empty_update,
            empty_score,
        )
        attached_score = torch.where(
            observed,
            ema * attached_score + update * attached_update,
            attached_score,
        )
        scene_core_score = torch.where(
            observed,
            ema * scene_core_score + update * scene_core_update,
            scene_core_score,
        )
        self.strategy_state["water_exposure_score"] = water_score
        self.strategy_state["scene_exposure_score"] = scene_score
        self.strategy_state["depth_edge_support_score"] = depth_edge_score
        self.strategy_state["color_residual_support_score"] = color_residual_score
        self.strategy_state["view_consistency_score"] = view_consistency_score
        self.strategy_state["empty_freewater_exposure_score"] = empty_score
        self.strategy_state["attached_veil_exposure_score"] = attached_score
        self.strategy_state["scene_core_exposure_score"] = scene_core_score
        self.water_exposure_score = water_score
        self.scene_exposure_score = scene_score
        self.depth_edge_support_score = depth_edge_score
        self.color_residual_support_score = color_residual_score
        self.view_consistency_score = view_consistency_score
        self.empty_freewater_exposure_score = empty_score
        self.attached_veil_exposure_score = attached_score
        self.scene_core_exposure_score = scene_core_score

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
        del n_before_refine
        if self._uses_legacy_water_lifecycle():
            self._ensure_water_lifecycle_buffers()
        if self._uses_surface_carrier():
            self.surface_carrier.sync_from_strategy_state(self.strategy_state)

    def _gaussian_responsibility_scores(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-Gaussian freewater / scene / haze scores from cross-view evidence."""
        self._ensure_water_lifecycle_buffers()
        water_score = self.strategy_state["water_exposure_score"].detach()
        scene_score = self.strategy_state["scene_exposure_score"].detach()
        edge_score = self.strategy_state["depth_edge_support_score"].detach()
        residual_score = self.strategy_state["color_residual_support_score"].detach()
        consistency_score = self.strategy_state["view_consistency_score"].detach()
        empty_score = self.strategy_state["empty_freewater_exposure_score"].detach()
        attached_score = self.strategy_state["attached_veil_exposure_score"].detach()
        scene_core_score = self.strategy_state["scene_core_exposure_score"].detach()
        empty_combined = torch.maximum(water_score, empty_score)
        attached_combined = torch.maximum(residual_score, attached_score)
        scene_combined = torch.maximum(scene_score, scene_core_score)
        total = empty_combined + scene_combined + attached_combined + 1e-6
        freewater_ratio = empty_combined / total
        scene_ratio = scene_combined / total
        haze_ratio = attached_combined / total
        edge_guard = (1.0 - edge_score / max(self.config.water_depth_edge_support_max, 1e-4)).clamp(0.0, 1.0)
        residual_guard = (1.0 - attached_combined / max(self.config.water_color_residual_support_max, 1e-4)).clamp(0.0, 1.0)
        consistency_guard = (0.25 + 0.75 * consistency_score).clamp(0.0, 1.0)
        freewater = (freewater_ratio * edge_guard * residual_guard * consistency_guard).clamp(0.0, 1.0)
        scene = torch.maximum(scene_ratio, 0.35 * (1.0 - freewater)).clamp(0.0, 1.0)
        haze = (haze_ratio * (1.0 - freewater)).clamp(0.0, 1.0)
        if self._uses_surface_carrier():
            self.surface_carrier.sync_from_strategy_state(self.strategy_state)
            ownership = self.surface_carrier.probabilities()
            certified = ownership["certified"]
            freewater = torch.where(certified, ownership["freewater"], freewater)
            scene = torch.where(certified, ownership["scene"], scene)
            haze = torch.where(certified, ownership["haze"], haze)
        return freewater.unsqueeze(-1), scene.unsqueeze(-1), haze.unsqueeze(-1)

    def _gaussian_null_dominance_score(self) -> torch.Tensor:
        """Return per-Gaussian freewater dominance used for hard blocks."""
        freewater, _, _ = self._gaussian_responsibility_scores()
        return freewater

    def _build_intrinsic_visibility_gate(
        self,
        alpha: torch.Tensor,
        water_probability: Optional[torch.Tensor] = None,
        alpha_water: Optional[torch.Tensor] = None,
        scene_rescue: Optional[torch.Tensor] = None,
        residual_water: Optional[torch.Tensor] = None,
        connected_freewater: Optional[torch.Tensor] = None,
        responsibility_freewater: Optional[torch.Tensor] = None,
        responsibility_haze: Optional[torch.Tensor] = None,
        responsibility_scene: Optional[torch.Tensor] = None,
        responsibility_uncertainty: Optional[torch.Tensor] = None,
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
        if responsibility_freewater is None:
            responsibility_freewater = torch.zeros_like(alpha)
        if responsibility_haze is None:
            responsibility_haze = residual_water
        if responsibility_scene is None:
            responsibility_scene = scene_rescue
        if responsibility_uncertainty is None:
            responsibility_uncertainty = torch.zeros_like(alpha)

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
        responsibility_freewater = torch.nan_to_num(
            responsibility_freewater.detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        responsibility_haze = torch.nan_to_num(
            responsibility_haze.detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        responsibility_scene = torch.nan_to_num(
            responsibility_scene.detach(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        responsibility_uncertainty = torch.nan_to_num(
            responsibility_uncertainty.detach(),
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
        freewater_visibility = 1.0 - torch.pow(
            (1.0 - responsibility_freewater).clamp(0.0, 1.0),
            max(self.config.responsibility_freewater_visibility_gamma, 1e-4),
        )
        haze_visibility = 1.0 - torch.pow(
            (1.0 - responsibility_haze).clamp(0.0, 1.0),
            max(self.config.responsibility_haze_visibility_gamma, 1e-4),
        )
        water_strength = torch.maximum(
            water_strength,
            residual_visibility * max(min(self.config.residual_water_visibility_strength, 1.0), 0.0),
        )
        water_strength = torch.maximum(
            water_strength,
            connected_visibility * max(min(self.config.connected_freewater_visibility_strength, 1.0), 0.0),
        )
        water_strength = torch.maximum(
            water_strength,
            freewater_visibility * max(min(self.config.responsibility_freewater_strength, 1.0), 0.0),
        )
        water_strength = torch.maximum(
            water_strength,
            haze_visibility * max(min(self.config.responsibility_haze_strength, 1.0), 0.0),
        )
        freewater_rescue_cancel = torch.maximum(
            residual_water,
            connected_freewater * 0.75,
        )
        freewater_rescue_cancel = torch.maximum(
            freewater_rescue_cancel,
            responsibility_freewater,
        )
        freewater_rescue_cancel = freewater_rescue_cancel * (
            1.0 - 0.75 * responsibility_scene
        ).clamp(0.0, 1.0)
        scene_rescue_safe = torch.maximum(scene_rescue, responsibility_scene) * (
            1.0 - self.config.residual_water_scene_suppression * freewater_rescue_cancel
        ).clamp(0.0, 1.0)
        rescue_strength = max(min(self.config.intrinsic_visibility_rescue_strength, 1.0), 0.0)
        water_strength = water_strength * (
            1.0 - self.config.responsibility_uncertainty_visibility_floor * responsibility_uncertainty
        ).clamp(0.0, 1.0)
        visibility = 1.0 - water_strength * (1.0 - rescue_strength * scene_rescue_safe)
        visibility_floor = max(min(self.config.intrinsic_visibility_floor, 1.0), 0.0) * scene_rescue_safe
        visibility_floor = torch.maximum(
            visibility_floor,
            max(min(self.config.responsibility_uncertainty_visibility_floor, 1.0), 0.0) * responsibility_uncertainty,
        )
        visibility = torch.maximum(visibility, visibility_floor)
        return visibility.clamp(0.0, 1.0)

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        black = background.new_zeros(height, width, 3)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        freewater = torch.ones_like(accumulation)
        water_background_image = rgb.clone()
        water_coefficients = background.new_zeros(*rgb.shape[:2], 3)
        return {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
            "background": background,
            "intrinsic_color_render": black,
            "intrinsic_color_render_raw": black.clone(),
            "intrinsic_visibility": accumulation.clone(),
            "water_accumulation": freewater,
            "rendered_null_probability": freewater.clone(),
            "rendered_scene_probability": accumulation.clone(),
            "rendered_haze_probability": accumulation.clone(),
            "water_background_image": water_background_image,
            "background_backscatter_coefficients": water_coefficients.clone(),
            "background_attenuation_coefficients": water_coefficients.clone(),
            "direct_transmittance": torch.ones_like(rgb),
            "optical_depth": accumulation.clone(),
            "water_calibration_ambient_delta": water_coefficients.clone(),
            "water_calibration_backscatter_delta": water_coefficients.clone(),
            "water_calibration_attenuation_delta": water_coefficients.clone(),
            "responsibility_freewater": freewater.clone(),
            "responsibility_haze": accumulation.clone(),
            "responsibility_scene": accumulation.clone(),
            "responsibility_uncertainty": accumulation.clone(),
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

    def _apply_water_formation_calibration(
        self,
        ambient_light_colors: torch.Tensor,
        backscatter_coefficients: torch.Tensor,
        attenuation_coefficients: torch.Tensor,
        calibration_raw: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply small physical residuals before underwater image formation."""
        zero_delta = torch.zeros(
            *ambient_light_colors.shape[:-1],
            9,
            device=ambient_light_colors.device,
            dtype=ambient_light_colors.dtype,
        )
        if (
            not self.config.enable_water_formation_calibrator
            or calibration_raw is None
            or calibration_raw.numel() == 0
        ):
            return ambient_light_colors, backscatter_coefficients, attenuation_coefficients, zero_delta

        ambient_delta_scale = max(float(self.config.water_calibrator_ambient_delta), 0.0)
        coeff_log_scale = max(float(self.config.water_calibrator_coeff_log_scale), 0.0)
        ambient_delta = calibration_raw[..., :3].to(ambient_light_colors) * ambient_delta_scale
        backscatter_log_delta = calibration_raw[..., 3:6].to(backscatter_coefficients) * coeff_log_scale
        attenuation_log_delta = calibration_raw[..., 6:9].to(attenuation_coefficients) * coeff_log_scale
        ambient_light_colors = torch.clamp(ambient_light_colors + ambient_delta, 0.0, 1.0)
        backscatter_coefficients = backscatter_coefficients * torch.exp(backscatter_log_delta)
        attenuation_coefficients = attenuation_coefficients * torch.exp(attenuation_log_delta)
        calibration_delta = torch.cat(
            [ambient_delta, backscatter_log_delta, attenuation_log_delta],
            dim=-1,
        )
        return ambient_light_colors, backscatter_coefficients, attenuation_coefficients, calibration_delta

    def _get_outputs_nextgen(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Render compositing-exact underwater and clean radiance in one pass."""
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if not self.training and self._uses_surface_carrier():
            self.surface_carrier.sync_from_strategy_state(self.strategy_state)
            self.surface_carrier.finalize_pending_window(
                means=self.means,
                opacity=torch.sigmoid(self.opacities).squeeze(-1),
            )
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        if self._uses_surface_carrier():
            self.surface_carrier.sync_from_strategy_state(self.strategy_state)
            carrier_probability = self.surface_carrier.probability().detach()
        else:
            carrier_probability = torch.ones(
                self.num_points, device=self.means.device, dtype=self.means.dtype
            )

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color.to(self.device),
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            means = self.means[crop_ids]
            scales = self.scales[crop_ids]
            quats = self.quats[crop_ids]
            opacities = self.opacities[crop_ids]
            features_dc = self.features_dc[crop_ids]
            features_rest = self.features_rest[crop_ids]
            carrier_probability = carrier_probability[crop_ids]
        else:
            means = self.means
            scales = self.scales
            quats = self.quats
            opacities = self.opacities
            features_dc = self.features_dc
            features_rest = self.features_rest

        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors = torch.sigmoid(colors).squeeze(1)
            sh_degree_to_use = None

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        intrinsics = camera.get_intrinsics_matrices().to(self.device)
        width, height = int(camera.width.item()), int(camera.height.item())
        self.last_size = (height, width)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError(f"Unknown rasterize_mode: {self.config.rasterize_mode}")

        camera_to_world = torch.inverse(viewmat)
        water_background = self.path_integrated_renderer.render_background(
            height=height,
            width=width,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            dtype=means.dtype,
        )
        raw_opacity = torch.sigmoid(opacities).squeeze(-1)
        effective_opacity = raw_opacity * carrier_probability
        # Training losses and metrics only consume underwater RGB, depth, and
        # alpha. Evaluation still composites both radiance layers exactly.
        render_intrinsic = not self.training

        def transform_colors(**projection) -> torch.Tensor:
            return self.path_integrated_renderer.transform_projected_colors(
                intrinsic_colors=projection["colors"],
                means=means,
                camera_to_world=camera_to_world,
                radii=projection["radii"],
                camera_ids=projection["camera_ids"],
                gaussian_ids=projection["gaussian_ids"],
                include_intrinsic=render_intrinsic,
            )

        render, alpha, self.info = rasterization(
            means=means,
            quats=quats,
            scales=torch.exp(scales),
            opacities=effective_opacity,
            colors=colors,
            viewmats=viewmat,
            Ks=intrinsics,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB+ED",
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
            color_transform=transform_colors,
            collect_contributions=False,
        )
        self._prepare_gaussian_densification_backward()

        rendered_underwater = render[..., :3]
        if render_intrinsic:
            intrinsic = render[..., 3:6].clamp(0.0, 1.0)
        else:
            # Preserve the output contract without retaining an unused raster
            # graph during training. Clear renders are produced in eval mode.
            intrinsic = torch.zeros_like(rendered_underwater)
        expected_depth = render[..., -1:]
        rgb = (
            rendered_underwater
            + (1.0 - alpha) * water_background["backscatter_radiance"]
        ).clamp(0.0, 1.0)
        transport_diagnostics = self.path_integrated_renderer.diagnostics(
            expected_depth=expected_depth,
            alpha=alpha,
            background=water_background,
        )
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], height, width)

        far_depth = self.config.water_transport_depth_scale * self.config.water_transport_far_distance
        depth = torch.where(
            alpha > 1e-5,
            torch.nan_to_num(expected_depth, nan=far_depth, posinf=far_depth, neginf=0.0),
            torch.full_like(expected_depth, far_depth),
        ).squeeze(0)
        background = self._get_background_color()
        if background.shape[0] == 3 and not self.training:
            background = background.expand(height, width, 3)

        zeros_rgb = torch.zeros_like(water_background["ambient"])
        water_probability = (1.0 - alpha).clamp(0.0, 1.0)
        self._last_carrier_camera_origin = camera_to_world[0, :3, 3].detach()
        return {
            "rgb": rgb.squeeze(0),
            "depth": depth,
            "accumulation": alpha.squeeze(0),
            "background": background,
            "intrinsic_color_render": intrinsic.squeeze(0),
            "intrinsic_color_render_raw": intrinsic.squeeze(0),
            "intrinsic_visibility": alpha.squeeze(0),
            "water_accumulation": water_probability.squeeze(0),
            "rendered_null_probability": water_probability.squeeze(0),
            "rendered_scene_probability": alpha.squeeze(0),
            "rendered_haze_probability": torch.zeros_like(alpha.squeeze(0)),
            "water_background_image": water_background["backscatter_radiance"].squeeze(0),
            "background_backscatter_coefficients": water_background["backscatter"].squeeze(0),
            "background_attenuation_coefficients": water_background["attenuation"].squeeze(0),
            "direct_transmittance": transport_diagnostics["direct_transmittance"].squeeze(0),
            "optical_depth": transport_diagnostics["optical_depth"].squeeze(0),
            "water_calibration_ambient_delta": zeros_rgb.squeeze(0),
            "water_calibration_backscatter_delta": zeros_rgb.squeeze(0),
            "water_calibration_attenuation_delta": zeros_rgb.squeeze(0),
        }

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if self._uses_nextgen_water_model():
            return self._get_outputs_nextgen(camera)
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if not self.training and self._uses_surface_carrier():
            self.surface_carrier.sync_from_strategy_state(self.strategy_state)
            self.surface_carrier.finalize_pending_window(
                means=self.means,
                opacity=torch.sigmoid(self.opacities).squeeze(-1),
            )

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

        direct_carrier_clear = self._uses_surface_carrier()
        if self.config.enable_water_null_occupancy and not direct_carrier_clear:
            (
                gaussian_freewater_scores,
                gaussian_scene_scores,
                gaussian_haze_scores,
            ) = self._gaussian_responsibility_scores()
        else:
            gaussian_freewater_scores = torch.zeros_like(self.opacities)
            gaussian_scene_scores = torch.ones_like(self.opacities)
            gaussian_haze_scores = torch.zeros_like(self.opacities)
        if direct_carrier_clear:
            self.surface_carrier.sync_from_strategy_state(self.strategy_state)
            carrier_render_weights = self.surface_carrier.render_weights()
            underwater_geometry_gate = carrier_render_weights["underwater_geometry"].unsqueeze(-1)
            clear_scene_gate = carrier_render_weights["clear_scene"].unsqueeze(-1)
            surface_fraction = carrier_render_weights["surface_fraction"].unsqueeze(-1)
            medium_fraction = carrier_render_weights["medium_fraction"].unsqueeze(-1)
        else:
            carrier_render_weights = None
            underwater_geometry_gate = torch.ones_like(self.opacities)
            clear_scene_gate = torch.ones_like(self.opacities)
            surface_fraction = torch.ones_like(self.opacities)
            medium_fraction = torch.zeros_like(self.opacities)
        if crop_ids is not None:
            gaussian_freewater_scores_crop = gaussian_freewater_scores[crop_ids]
            gaussian_scene_scores_crop = gaussian_scene_scores[crop_ids]
            gaussian_haze_scores_crop = gaussian_haze_scores[crop_ids]
            underwater_geometry_gate_crop = underwater_geometry_gate[crop_ids]
            clear_scene_gate_crop = clear_scene_gate[crop_ids]
            surface_fraction_crop = surface_fraction[crop_ids]
            medium_fraction_crop = medium_fraction[crop_ids]
        else:
            gaussian_freewater_scores_crop = gaussian_freewater_scores
            gaussian_scene_scores_crop = gaussian_scene_scores
            gaussian_haze_scores_crop = gaussian_haze_scores
            underwater_geometry_gate_crop = underwater_geometry_gate
            clear_scene_gate_crop = clear_scene_gate
            surface_fraction_crop = surface_fraction
            medium_fraction_crop = medium_fraction

        total_opacity_crop = torch.sigmoid(opacities_crop)
        if direct_carrier_clear:
            surface_opacity_crop, _ = self.surface_carrier.split_optical_thickness(
                total_opacity_crop, surface_fraction_crop
            )
        else:
            surface_opacity_crop = total_opacity_crop

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
        
        # Query the validated WPP for the image background. Gaussian properties
        # are queried after projection/SH evaluation by color_transform below.
        camtoworlds = torch.inverse(viewmat)  # [C, 4, 4]
        self._last_carrier_camera_origin = camtoworlds[0, :3, 3].detach()

        def _query_base_water_properties(
            directions: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if directions.numel() == 0:
                empty = directions.new_zeros((0, 3))
                return empty, empty, empty, directions.new_zeros((0, 9))
            encoded = self.line_of_sight_direction_encoding(directions)
            if self.config.mlp_type == "tcnn":
                raw = self.water_properties_predictor(encoded)
            else:
                raw = self.water_properties_predictor(encoded.float())
            calibration_raw = None
            if self.config.enable_water_formation_calibrator:
                calibration_raw = self.water_formation_calibrator(encoded)
            ambient = self.ambient_light_activation(raw[..., :3])
            backscatter = self.water_coefficient_activation(raw[..., 3:6])
            attenuation = self.water_coefficient_activation(raw[..., 6:9])
            return self._apply_water_formation_calibration(
                ambient,
                backscatter,
                attenuation,
                calibration_raw,
            )

        y = torch.arange(H, device=self.device, dtype=K.dtype) + 0.5
        x = torch.arange(W, device=self.device, dtype=K.dtype) + 0.5
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
        (
            pixel_ambient_light_colors,
            pixel_backscatter_coefficients,
            pixel_attenuation_coefficients,
            pixel_water_calibration_delta,
        ) = _query_base_water_properties(pixel_line_of_sight_directions_flat)
        pixel_ambient_light_colors = pixel_ambient_light_colors.view(*image_shape, 3)
        pixel_backscatter_coefficients = pixel_backscatter_coefficients.view(*image_shape, 3)
        pixel_attenuation_coefficients = pixel_attenuation_coefficients.view(*image_shape, 3)
        pixel_water_calibration_delta = pixel_water_calibration_delta.view(*image_shape, 9)
        field_enabled = (
            self.config.enable_path_integrated_transport
            and self.step >= self.config.water_field_start_step
        )
        transport_prefix = None
        if field_enabled:
            transport_prefix = self.path_integrated_renderer.build_frustum_prefix(
                height=H,
                width=W,
                intrinsics=K,
                camera_to_world=camtoworlds,
                dtype=means_crop.dtype,
                query_base_properties=_query_base_water_properties,
                enabled=True,
                base_property_images=(
                    pixel_ambient_light_colors,
                    pixel_backscatter_coefficients,
                    pixel_attenuation_coefficients,
                ),
            )

        def _projection_water_transform(**projection) -> torch.Tensor:
            intrinsic_colors = projection["colors"]
            radii = projection["radii"]
            projected_means = projection["means2d"]
            projected_depths = projection["depths"]
            if projection.get("camera_ids") is not None or projection.get("gaussian_ids") is not None:
                raise NotImplementedError("Packed SeaFree transport is not supported")
            transformed = []
            depth_scale = max(float(self.config.water_transport_depth_scale), 1e-4)
            for camera_index in range(intrinsic_colors.shape[0]):
                intrinsic = intrinsic_colors[camera_index]
                visible_ids = torch.where(radii[camera_index] > 0)[0]
                degraded = intrinsic.clone()
                if visible_ids.numel() > 0:
                    if field_enabled:
                        assert transport_prefix is not None
                        origin = camtoworlds[camera_index, :3, 3].detach().to(means_crop)
                        positions = means_crop[visible_ids].detach()
                        vectors = positions - origin
                        distances = torch.linalg.vector_norm(vectors, dim=-1).clamp_min(1e-5)
                        directions = vectors / distances[:, None]
                        ambient, backscatter, attenuation, _ = _query_base_water_properties(
                            directions
                        )
                        path = self.path_integrated_renderer.sample_projected_paths(
                            prefix=transport_prefix,
                            means2d=projected_means[camera_index],
                            depths=projected_depths[camera_index],
                            visible_ids=visible_ids,
                            base_ambient=ambient,
                            base_backscatter=backscatter,
                            base_attenuation=attenuation,
                            ray_directions=directions,
                            path_distances=distances,
                        )
                        direct = path["direct_transmittance"]
                        backscatter_radiance = path["backscatter_radiance"]
                    else:
                        origin = camtoworlds[camera_index, :3, 3].detach().to(means_crop)
                        positions = means_crop[visible_ids].detach()
                        vectors = positions - origin
                        distances = torch.linalg.vector_norm(vectors, dim=-1).clamp_min(1e-5)
                        directions = vectors / distances[:, None]
                        ambient, backscatter, attenuation, _ = _query_base_water_properties(
                            directions
                        )
                        normalized_distance = (distances / depth_scale).clamp(
                            0.0, float(self.config.water_transport_far_distance)
                        )[:, None]
                        direct = torch.exp(-attenuation * normalized_distance)
                        backscatter_radiance = ambient * (
                            1.0 - torch.exp(-backscatter * normalized_distance)
                        )
                    surface_radiance = intrinsic[visible_ids] * direct + backscatter_radiance
                    surface_mix = surface_fraction_crop[visible_ids].to(surface_radiance)
                    medium_mix = medium_fraction_crop[visible_ids].to(surface_radiance)
                    degraded_visible = (
                        surface_mix * surface_radiance + medium_mix * backscatter_radiance
                    )
                    degraded = degraded.index_copy(0, visible_ids, degraded_visible)

                def responsibility_channel(channel: torch.Tensor) -> torch.Tensor:
                    value = channel.to(intrinsic)
                    if value.dim() == 1:
                        value = value[:, None]
                    return value

                channels = [degraded, intrinsic]
                if direct_carrier_clear and not self.training:
                    channels.extend(
                        [
                            responsibility_channel(medium_fraction_crop),
                        ]
                    )
                else:
                    channels.extend(
                        [
                            responsibility_channel(gaussian_freewater_scores_crop),
                            responsibility_channel(gaussian_scene_scores_crop),
                            responsibility_channel(gaussian_haze_scores_crop),
                        ]
                    )
                transformed.append(torch.cat(channels, dim=-1))
            return torch.stack(transformed, dim=0)

        background_transport = self.path_integrated_renderer.integrate_background_from_base(
            ambient=pixel_ambient_light_colors,
            backscatter=pixel_backscatter_coefficients,
            attenuation=pixel_attenuation_coefficients,
            directions=pixel_line_of_sight_directions,
            camera_origin=camtoworlds[0, :3, 3].detach(),
            enabled=field_enabled,
            prefix=transport_prefix,
        )
        water_background_image = background_transport["backscatter_radiance"]
        background_attenuation_coefficients = background_transport["attenuation"]
        background_backscatter_coefficients = background_transport["backscatter"]

        # render [C,H,W,D] alpha [1,H,W,1]
        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=(total_opacity_crop * underwater_geometry_gate_crop).squeeze(-1),
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
            color_transform=_projection_water_transform,
            collect_contributions=False,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        self._prepare_gaussian_densification_backward()
        alpha = alpha[:, ...]
        background = self._get_background_color()

        rgb = render[:, ..., :3] + (1 - alpha) * water_background_image
        rgb = torch.clamp(rgb, 0.0, 1.0)

        intrinsic_color_render_raw = torch.clamp(render[:, ..., 3:6], 0.0, 1.0)
        carrier_clear_render = None
        if direct_carrier_clear and not self.training:
            # A separate opacity stream is required for correct clear-view
            # transmittance. Reusing underwater alpha would leave medium splats
            # as black occluders in front of real surfaces. This extra raster is
            # evaluation-only and therefore does not change training time/memory.
            with torch.no_grad():
                clear_render, _, _ = rasterization(
                    means=means_crop,
                    quats=quats_crop,
                    scales=torch.exp(scales_crop),
                    opacities=(
                            surface_opacity_crop
                    ).squeeze(-1),
                    colors=colors_crop,
                    viewmats=viewmat,
                    Ks=K,
                    width=W,
                    height=H,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1e10,
                    render_mode="RGB",
                    sh_degree=sh_degree_to_use,
                    sparse_grad=False,
                    absgrad=False,
                    rasterize_mode=self.config.rasterize_mode,
                )
                carrier_clear_render = clear_render[..., :3].clamp(0.0, 1.0)
        if direct_carrier_clear:
            if self.training:
                # Keep the training raster payload identical to the baseline.
                # Carrier diagnostics come directly from persistent 3D state.
                rendered_null_probability = torch.zeros_like(alpha)
                rendered_scene_probability = torch.ones_like(alpha)
                water_accumulation = torch.zeros_like(alpha)
            else:
                rendered_null_probability = (
                    render[:, ..., 6:7] / alpha.clamp_min(1e-6)
                ).clamp(0.0, 1.0)
                rendered_scene_probability = (1.0 - rendered_null_probability).clamp(0.0, 1.0)
                water_accumulation = render[:, ..., 6:7].clamp(0.0, 1.0)
            rendered_haze_probability = torch.zeros_like(alpha)
            intrinsic_visibility = torch.ones_like(alpha)
            intrinsic_color_render_clamped = (
                carrier_clear_render
                if carrier_clear_render is not None
                else intrinsic_color_render_raw
            )
        else:
            water_accumulation = render[:, ..., 6:7].clamp(0.0, 1.0)
            scene_accumulation = render[:, ..., 7:8].clamp(0.0, 1.0)
            haze_accumulation = render[:, ..., 8:9].clamp(0.0, 1.0)
            rendered_null_probability = (
                water_accumulation / alpha.clamp_min(1e-6)
            ).clamp(0.0, 1.0)
            rendered_scene_probability = (
                scene_accumulation / alpha.clamp_min(1e-6)
            ).clamp(0.0, 1.0)
            rendered_haze_probability = (
                haze_accumulation / alpha.clamp_min(1e-6)
            ).clamp(0.0, 1.0)
            intrinsic_visibility = self._build_intrinsic_visibility_gate(
                alpha,
                rendered_null_probability,
                water_accumulation,
                responsibility_freewater=rendered_null_probability,
                responsibility_scene=rendered_scene_probability,
                responsibility_haze=rendered_haze_probability,
            )
            intrinsic_color_render_clamped = (
                intrinsic_color_render_raw * intrinsic_visibility
            )


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

        ambient_delta_scale = max(float(self.config.water_calibrator_ambient_delta), 1e-6)
        coeff_log_scale = max(float(self.config.water_calibrator_coeff_log_scale), 1e-6)
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
            "rendered_scene_probability": rendered_scene_probability.squeeze(0),  # type: ignore
            "rendered_haze_probability": rendered_haze_probability.squeeze(0),  # type: ignore
            "water_background_image": water_background_image,
            "background_backscatter_coefficients": background_backscatter_coefficients,
            "background_attenuation_coefficients": background_attenuation_coefficients,
            "water_calibration_ambient_delta": (
                pixel_water_calibration_delta[..., :3] / ambient_delta_scale * 0.5 + 0.5
            ).clamp(0.0, 1.0),
            "water_calibration_backscatter_delta": (
                pixel_water_calibration_delta[..., 3:6] / coeff_log_scale * 0.5 + 0.5
            ).clamp(0.0, 1.0),
            "water_calibration_attenuation_delta": (
                pixel_water_calibration_delta[..., 6:9] / coeff_log_scale * 0.5 + 0.5
            ).clamp(0.0, 1.0),
            "water_calibration_delta_raw": pixel_water_calibration_delta,
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

    def _get_metrics_dict_nextgen(self, outputs, batch) -> Dict[str, torch.Tensor]:
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        if self._uses_surface_carrier():
            carrier_probability_mean = self.surface_carrier.probability().mean()
            carrier_certified_fraction = self.surface_carrier.probabilities()[
                "certified"
            ].float().mean()
        else:
            carrier_probability_mean = torch.ones((), device=self.device)
            carrier_certified_fraction = torch.zeros((), device=self.device)
        metrics_dict: Dict[str, torch.Tensor] = {
            "psnr": self.psnr(outputs["rgb"], gt_rgb),
            "gaussian_count": torch.tensor(float(self.num_points), device=self.device),
            "carrier_probability_mean": carrier_probability_mean,
            "carrier_certified_fraction": carrier_certified_fraction,
        }
        if self.config.color_corrected_metrics:
            metrics_dict["cc_psnr"] = self.psnr(color_correct(outputs["rgb"], gt_rgb), gt_rgb)
        for channel in range(3):
            attenuation = outputs["background_attenuation_coefficients"][..., channel]
            backscatter = outputs["background_backscatter_coefficients"][..., channel]
            ambient = outputs["water_background_image"][..., channel]
            metrics_dict[f"background_attenuation_coefficient_{channel}_mean"] = attenuation.mean()
            metrics_dict[f"background_backscatter_coefficient_{channel}_mean"] = backscatter.mean()
            metrics_dict[f"water_background_color_{channel}_mean"] = ambient.mean()
            metrics_dict[f"background_attenuation_coefficient_{channel}_max"] = attenuation.max()
            metrics_dict[f"background_backscatter_coefficient_{channel}_max"] = backscatter.max()
            metrics_dict[f"water_background_color_{channel}_max"] = ambient.max()
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        if self._uses_nextgen_water_model():
            return self._get_metrics_dict_nextgen(outputs, batch)
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
            metrics_dict["water_null_probability_mean"] = self.last_water_probability_mean.to(self.device)
            metrics_dict["water_pure_probability_mean"] = self.last_water_pure_probability_mean.to(self.device)
            metrics_dict["water_transition_probability_mean"] = self.last_water_transition_probability_mean.to(self.device)
            metrics_dict["water_null_strict_probability_mean"] = self.last_water_strict_probability_mean.to(self.device)
            metrics_dict["water_horizon_probability_mean"] = self.last_water_horizon_probability_mean.to(self.device)
            metrics_dict["residual_water_probability_mean"] = self.last_residual_water_probability_mean.to(self.device)
            metrics_dict["connected_freewater_probability_mean"] = self.last_connected_freewater_probability_mean.to(self.device)
            metrics_dict["anchored_freewater_probability_mean"] = self.last_anchored_freewater_probability_mean.to(self.device)
            metrics_dict["anchored_haze_probability_mean"] = self.last_anchored_haze_probability_mean.to(self.device)
            metrics_dict["water_anchor_similarity_mean"] = self.last_water_anchor_similarity_mean.to(self.device)
            metrics_dict["water_matte_scene_protection_mean"] = self.last_water_matte_scene_protection_mean.to(self.device)
            metrics_dict["empty_freewater_probability_mean"] = self.last_empty_freewater_probability_mean.to(self.device)
            metrics_dict["attached_veil_probability_mean"] = self.last_attached_veil_probability_mean.to(self.device)
            metrics_dict["scene_core_probability_mean"] = self.last_scene_core_probability_mean.to(self.device)
            metrics_dict["veil_alpha_mean"] = self.last_veil_alpha_mean.to(self.device)
            metrics_dict["responsibility_freewater_mean"] = self.last_responsibility_freewater_mean.to(self.device)
            metrics_dict["responsibility_haze_mean"] = self.last_responsibility_haze_mean.to(self.device)
            metrics_dict["responsibility_scene_mean"] = self.last_responsibility_scene_mean.to(self.device)
            metrics_dict["responsibility_uncertainty_mean"] = self.last_responsibility_uncertainty_mean.to(self.device)
            metrics_dict["water_calibration_delta_mean"] = self.last_water_calibration_delta_mean.to(self.device)
            if self._uses_legacy_water_lifecycle():
                self._ensure_water_lifecycle_buffers()
                metrics_dict["water_exposure_score_mean"] = self.strategy_state["water_exposure_score"].mean()
                metrics_dict["scene_exposure_score_mean"] = self.strategy_state["scene_exposure_score"].mean()
                metrics_dict["depth_edge_support_score_mean"] = self.strategy_state["depth_edge_support_score"].mean()
                metrics_dict["color_residual_support_score_mean"] = self.strategy_state["color_residual_support_score"].mean()
                metrics_dict["view_consistency_score_mean"] = self.strategy_state["view_consistency_score"].mean()
                metrics_dict["empty_freewater_exposure_score_mean"] = self.strategy_state["empty_freewater_exposure_score"].mean()
                metrics_dict["attached_veil_exposure_score_mean"] = self.strategy_state["attached_veil_exposure_score"].mean()
                metrics_dict["scene_core_exposure_score_mean"] = self.strategy_state["scene_core_exposure_score"].mean()
            metrics_dict["water_no_grow_candidates"] = torch.tensor(
                float(self.last_water_no_grow_candidates),
                device=self.device,
            )
            metrics_dict["water_prune_candidates"] = torch.tensor(
                float(self.last_water_prune_candidates),
                device=self.device,
            )
        if self._uses_surface_carrier():
            carrier_ownership = self.surface_carrier.probabilities()
            carrier_render_weights = self.surface_carrier.render_weights()
            metrics_dict["carrier_freewater_mean"] = carrier_ownership["freewater"].mean()
            metrics_dict["carrier_haze_mean"] = carrier_ownership["haze"].mean()
            metrics_dict["carrier_scene_mean"] = carrier_ownership["scene"].mean()
            metrics_dict["carrier_certified_fraction"] = carrier_ownership[
                "certified"
            ].float().mean()
            metrics_dict["carrier_certified_medium_fraction"] = carrier_ownership[
                "certified_medium"
            ].float().mean()
            metrics_dict["carrier_certified_surface_fraction"] = carrier_ownership[
                "certified_surface"
            ].float().mean()
            metrics_dict["carrier_observation_mass_mean"] = (
                self.surface_carrier.observation_mass.mean()
            )
            metrics_dict["carrier_epoch_distinct_views_mean"] = (
                self.surface_carrier.distinct_views.float().mean()
            )
            metrics_dict["carrier_lifetime_distinct_views_mean"] = (
                self.surface_carrier.lifetime_coverage.float().mean()
            )
            metrics_dict["carrier_window_parallax_mean"] = (
                self.surface_carrier.parallax_evidence.float().mean()
            )
            metrics_dict["carrier_medium_optical_fraction_mean"] = carrier_render_weights[
                "medium_fraction"
            ].mean()
            metrics_dict["carrier_topology_confidence_mean"] = (
                self.surface_carrier.topology_confidence.float().mean()
            )
            metrics_dict["carrier_adaptive_medium_anchor"] = (
                self.surface_carrier.adaptive_medium_anchor.float()
            )
        if self.config.enable_path_integrated_transport:
            field = self.path_integrated_renderer.field
            metrics_dict["water_field_plane_abs_mean"] = field.volume.abs().mean()
            metrics_dict["water_field_plane_abs_max"] = field.volume.abs().max()
            metrics_dict["water_field_anchor_abs_mean"] = field.latent_running_mean.abs().mean()
            metrics_dict["water_field_free_space_support_mean"] = (
                field.free_space_support.float().mean()
            )
            metrics_dict["water_field_free_space_support_min"] = (
                field.free_space_support.float().amin()
            )
            metrics_dict["water_field_support_observation_mass"] = (
                field.support_observation_mass.float()
            )
            metrics_dict["water_field_medium_optical_mass_mean"] = (
                field.medium_optical_mass.float().mean()
            )
            metrics_dict["water_field_ambient_observation_mass"] = (
                field.ambient_observation_mass.float()
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

    def _get_loss_dict_nextgen(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Original SeaFree supervision on the compact, GPU-only forward path."""
        gt_underwater = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        rendered_underwater = outputs["rgb"]
        pseudo_depth = self._downscale_if_required(batch["depth_image"]).to(self.device)
        pseudo_depth = pseudo_depth / pseudo_depth.amax().clamp_min(1e-6)
        rendered_depth = outputs["depth"]
        valid_image_mask = None

        if "mask" in batch:
            valid_image_mask = self._downscale_if_required(batch["mask"]).to(self.device)
            assert valid_image_mask.shape[:2] == gt_underwater.shape[:2] == rendered_underwater.shape[:2]
            gt_underwater = gt_underwater * valid_image_mask
            rendered_underwater = rendered_underwater * valid_image_mask
            pseudo_depth = pseudo_depth * valid_image_mask
            if rendered_depth is not None:
                rendered_depth = rendered_depth * valid_image_mask

        foreground_mask = (pseudo_depth > 1e-2).to(pseudo_depth.dtype)
        if foreground_mask.dim() == 2:
            foreground_mask = foreground_mask.unsqueeze(-1)
        background_mask = 1.0 - foreground_mask
        background_fraction = background_mask.mean()

        reconstruction_weight = 1.0 / (rendered_underwater.detach() + 1e-3)
        reconstruction_weight = torch.where(
            foreground_mask < 0.5,
            torch.ones_like(reconstruction_weight),
            reconstruction_weight,
        )
        weighted_l1 = torch.abs(
            (gt_underwater - rendered_underwater) * reconstruction_weight
        ).mean()
        gt_chw = gt_underwater.permute(2, 0, 1).unsqueeze(0)
        rendered_chw = rendered_underwater.permute(2, 0, 1).unsqueeze(0)
        weight_chw = reconstruction_weight.permute(2, 0, 1).unsqueeze(0)
        if gt_chw.shape[-1] > 800 or gt_chw.shape[-2] > 800:
            weighted_dssim = 1.0 - self.split_and_calculate_ssim(
                self.ssim, gt_chw * weight_chw, rendered_chw * weight_chw
            )
        else:
            weighted_dssim = 1.0 - self.ssim(
                gt_chw * weight_chw, rendered_chw * weight_chw
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
            if self.step < self.config.reset_alpha_every * self.config.refine_every + 200:
                scale_reg = 0.1 * scale_reg.mean()
            else:
                scale_reg = 100.0 * scale_reg.mean()
        else:
            scale_reg = torch.zeros((), device=self.device)

        depth_loss = torch.zeros((), device=self.device)
        if self.config.enable_coarse_grained_depth_loss and rendered_depth is not None:
            approximate_disparity = 1.0 / (rendered_depth.flatten() * 10.0 + 1.0)
            depth_loss = 1.0 - pearson_corrcoef(pseudo_depth.flatten(), approximate_disparity)
            depth_loss = torch.nan_to_num(depth_loss, nan=0.0, posinf=0.0, neginf=0.0)

        background_supervision = torch.zeros((), device=self.device)
        if self.config.enable_background_water_supervision and self.step < 15000:
            ambient = outputs["water_background_image"]
            ambient_weight = 1.0 / (ambient.detach() + 1e-3)
            background_supervision = self._weighted_mean(
                torch.abs((ambient - gt_underwater) * ambient_weight),
                background_mask.expand_as(ambient),
            )
            background_supervision = background_supervision * (background_fraction > 0.05).to(
                background_supervision.dtype
            )

        loss_dict = {
            "content_based_reconstruction_loss": (
                (1.0 - self.config.ssim_lambda) * weighted_l1
                + self.config.ssim_lambda * weighted_dssim
                + 0.01 * background_supervision
            ),
            "scale_regularization_loss": scale_reg,
            "coarse_grained_depth_loss": 0.1 * depth_loss,
        }
        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10.0 * total_variation_loss(self.bil_grids.grids)
        return loss_dict
    

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        if self._uses_nextgen_water_model():
            return self._get_loss_dict_nextgen(outputs, batch)
        gt_underwater_image = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        rendered_underwater_image = outputs["rgb"]
        render_mask = None


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
            render_mask = mask
            gt_underwater_image = gt_underwater_image * mask
            rendered_underwater_image = rendered_underwater_image * mask
            pseudo_depth = pseudo_depth * mask
            if rendered_depth is not None:
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
            freewater_responsibility = water_probabilities["responsibility_freewater"]
            haze_responsibility = water_probabilities["responsibility_haze"]
            scene_responsibility = water_probabilities["responsibility_scene"]
            uncertainty_responsibility = water_probabilities["responsibility_uncertainty"]
            strict_pure_water_probability = self._build_strict_water_null_probability(
                water_pure_probability,
                self.config.water_null_strict_threshold,
                self.config.water_pure_max_fraction,
            )
            strict_freewater_responsibility = self._build_strict_water_null_probability(
                water_probabilities["empty_freewater"],
                self.config.layer_empty_freewater_strict_threshold,
                self.config.responsibility_freewater_max_fraction,
            )
            strict_haze_responsibility = self._build_strict_water_null_probability(
                haze_responsibility,
                self.config.responsibility_strict_haze_threshold,
                self.config.responsibility_haze_max_fraction,
            )
            strict_transition_water_probability = torch.maximum(
                self._build_strict_water_null_probability(
                    water_transition_probability,
                    self.config.water_transition_strict_threshold,
                    self.config.water_transition_max_fraction,
                ),
                strict_haze_responsibility,
            )
            strict_destructive_water_probability = torch.maximum(
                strict_pure_water_probability,
                strict_freewater_responsibility,
            )
            strict_water_null_probability = torch.maximum(
                strict_destructive_water_probability,
                strict_transition_water_probability,
            )
            self.last_water_probability_mean = water_null_probability.mean().detach()
            self.last_water_pure_probability_mean = water_pure_probability.mean().detach()
            self.last_water_transition_probability_mean = water_transition_probability.mean().detach()
            self.last_water_strict_probability_mean = strict_water_null_probability.mean().detach()
            self.last_water_horizon_probability_mean = water_probabilities["horizon"].mean().detach()
            self.last_residual_water_probability_mean = water_probabilities["residual_water"].mean().detach()
            self.last_connected_freewater_probability_mean = water_probabilities["connected_freewater"].mean().detach()
            self.last_anchored_freewater_probability_mean = water_probabilities["anchored_freewater"].mean().detach()
            self.last_anchored_haze_probability_mean = water_probabilities["anchored_haze"].mean().detach()
            self.last_water_anchor_similarity_mean = water_probabilities["water_anchor_similarity"].mean().detach()
            self.last_water_matte_scene_protection_mean = water_probabilities["water_matte_scene_protection"].mean().detach()
            self.last_empty_freewater_probability_mean = water_probabilities["empty_freewater"].mean().detach()
            self.last_attached_veil_probability_mean = water_probabilities["attached_veil"].mean().detach()
            self.last_scene_core_probability_mean = water_probabilities["scene_core"].mean().detach()
            self.last_veil_alpha_mean = water_probabilities["veil_alpha"].mean().detach()
            self.last_responsibility_freewater_mean = freewater_responsibility.mean().detach()
            self.last_responsibility_haze_mean = haze_responsibility.mean().detach()
            self.last_responsibility_scene_mean = scene_responsibility.mean().detach()
            self.last_responsibility_uncertainty_mean = uncertainty_responsibility.mean().detach()
            self.last_water_calibration_delta_mean = outputs.get(
                "water_calibration_delta_raw",
                torch.zeros_like(outputs["water_background_image"]),
            ).abs().mean().detach()
            rendered_error = torch.abs(
                rendered_underwater_image.detach() - gt_underwater_image.detach()
            ).mean(dim=-1, keepdim=True)
            water_only_error = torch.abs(
                outputs["water_background_image"].detach() - gt_underwater_image.detach()
            ).mean(dim=-1, keepdim=True)
            water_only_advantage = torch.sigmoid(
                (rendered_error - water_only_error) / 0.04
            )
            trusted_freewater = torch.maximum(
                freewater_responsibility,
                water_probabilities["empty_freewater"],
            ) * (1.0 - scene_responsibility).clamp(0.0, 1.0)
            self.path_integrated_renderer.field.observe_freewater(
                gt_underwater_image,
                trusted_freewater,
            )
            self._update_water_lifecycle_scores(
                water_probability=freewater_responsibility,
                scene_probability=scene_responsibility,
                haze_probability=haze_responsibility,
                depth_edge_support=water_probabilities["depth_edge_support"],
                color_residual_support=water_probabilities["color_residual_support"],
                empty_freewater_probability=water_probabilities["empty_freewater"],
                attached_veil_probability=water_probabilities["attached_veil"],
                scene_core_probability=water_probabilities["scene_core"],
                water_similarity_probability=water_probabilities["water_anchor_similarity"],
                low_structure_probability=water_probabilities["water_matte_low_structure"],
                water_only_advantage=water_only_advantage,
                pseudo_depth=pseudo_depth,
                rendered_depth=rendered_depth,
                view_id=torch.as_tensor(batch["image_idx"], device=self.device),
                valid_image_mask=render_mask,
            )
        else:
            freewater_responsibility = torch.zeros_like(rendered_underwater_image[..., :1])
            haze_responsibility = torch.zeros_like(rendered_underwater_image[..., :1])
            scene_responsibility = torch.ones_like(rendered_underwater_image[..., :1])
            uncertainty_responsibility = torch.zeros_like(rendered_underwater_image[..., :1])

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
        if self.config.enable_coarse_grained_depth_loss and rendered_depth is not None:
            pseudo_depth_flattened = pseudo_depth.flatten()
            rendered_depth_flattened = rendered_depth.flatten()
            approximate_rendered_disparity = 1 / (rendered_depth_flattened * 10 + 1)
            coarse_grained_depth_loss += 1 - pearson_corrcoef(pseudo_depth_flattened, approximate_rendered_disparity)

        water_background_image = outputs["water_background_image"]
        
        background_water_supervision_loss = torch.tensor(0.0, device=self.device)
        if self.config.enable_background_water_supervision and self.step < 15000:
            strict_background_active = torch.zeros((), device=self.device)
            if self.config.enable_water_null_occupancy and strict_destructive_water_probability is not None:
                strict_background_active = (
                    (strict_destructive_water_probability > 0).float().mean() > 0.005
                ).to(water_background_image)
                background_ambient_light_weight = 1 / (
                    water_background_image.detach() + 1e-3
                )
                strict_background_loss = self._weighted_mean(
                    torch.abs(
                        (water_background_image - gt_underwater_image)
                        * background_ambient_light_weight
                    ),
                    strict_destructive_water_probability.repeat(1, 1, 3),
                )
                background_water_supervision_loss += (
                    strict_background_active * strict_background_loss
                )
            fallback_background_active = (
                (background_pixel_ratio > 0.05).to(water_background_image)
                * (1.0 - strict_background_active)
            )
            fallback_weight = (1.0 - foreground_mask).repeat(1, 1, 3)
            fallback_background_loss = self._weighted_mean(
                torch.abs(water_background_image - gt_underwater_image)
                / (water_background_image.detach() + 1e-3),
                fallback_weight,
            )
            background_water_supervision_loss += (
                fallback_background_active * fallback_background_loss
            )

        water_empty_occupancy_loss = torch.tensor(0.0, device=self.device)
        water_transition_alpha_loss = torch.tensor(0.0, device=self.device)
        water_intrinsic_null_loss = torch.tensor(0.0, device=self.device)
        water_background_ownership_loss = torch.tensor(0.0, device=self.device)
        residual_water_chroma_loss = torch.tensor(0.0, device=self.device)
        responsibility_scene_fidelity_loss = torch.tensor(0.0, device=self.device)
        responsibility_haze_fidelity_loss = torch.tensor(0.0, device=self.device)
        responsibility_boundary_fidelity_loss = torch.tensor(0.0, device=self.device)
        layer_empty_black_loss = torch.tensor(0.0, device=self.device)
        layer_attached_chroma_loss = torch.tensor(0.0, device=self.device)
        layer_detail_preservation_loss = torch.tensor(0.0, device=self.device)
        layer_veil_smoothness_loss = torch.tensor(0.0, device=self.device)
        layer_exclusivity_loss = torch.tensor(0.0, device=self.device)
        water_calibration_l2_loss = outputs.get(
            "water_calibration_delta_raw",
            torch.zeros_like(outputs["water_background_image"]),
        ).pow(2).mean()
        if (
            self.config.enable_water_null_occupancy
            and strict_destructive_water_probability is not None
            and self.step >= self.config.water_null_start_step
        ):
            water_empty_occupancy_loss = self._weighted_mean(
                outputs["accumulation"],
                strict_destructive_water_probability,
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
            layered_clean_outputs = self._compose_layered_clean_render(
                intrinsic_raw_for_chroma,
                water_background_image,
                water_probabilities,
            )
            layered_clean_render = layered_clean_outputs["clean"]
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
                haze_responsibility,
            )
            residual_chroma_weight = torch.maximum(
                residual_chroma_weight,
                water_probabilities["connected_freewater"] * 0.25,
            )
            residual_chroma_weight = torch.maximum(
                residual_chroma_weight,
                water_probabilities["anchored_haze"],
            )
            residual_chroma_weight = torch.maximum(
                residual_chroma_weight,
                water_probabilities["anchored_freewater"] * 0.35,
            )
            residual_chroma_weight = residual_chroma_weight * (1.0 - 0.55 * scene_responsibility).clamp(0.10, 1.0)
            residual_water_chroma_loss = self._weighted_mean(water_chroma_residual, residual_chroma_weight)

            layer_empty_black_loss = self._weighted_mean(
                torch.abs(layered_clean_render),
                strict_destructive_water_probability.repeat(1, 1, 3),
            )
            layered_blue_residual = torch.relu(
                layered_clean_render[..., 2:3]
                - torch.maximum(layered_clean_render[..., 0:1], layered_clean_render[..., 1:2])
                - self.config.intrinsic_water_blue_margin * 0.65
            )
            layered_cyan_residual = torch.relu(
                torch.minimum(layered_clean_render[..., 1:2], layered_clean_render[..., 2:3])
                - layered_clean_render[..., 0:1]
                - self.config.intrinsic_water_red_deficit_margin * 0.65
            )
            layer_chroma_residual = torch.maximum(layered_blue_residual, 0.75 * layered_cyan_residual)
            attached_weight = water_probabilities["attached_veil"] * (1.0 - water_probabilities["empty_freewater"]).clamp(0.0, 1.0)
            layer_attached_chroma_loss = self._weighted_mean(layer_chroma_residual, attached_weight)

            raw_detail = self._image_gradient_magnitude(intrinsic_raw_for_chroma)
            clean_detail = self._image_gradient_magnitude(layered_clean_render)
            detail_weight = attached_weight * (0.35 + 0.65 * water_probabilities["scene_core"]).clamp(0.0, 1.0)
            layer_detail_preservation_loss = self._weighted_mean(
                torch.relu(raw_detail * 0.60 - clean_detail),
                detail_weight,
            )
            veil_alpha_grad = self._image_gradient_magnitude(water_probabilities["veil_alpha"].repeat(1, 1, 3))
            smooth_weight = (
                1.0
                - torch.maximum(
                    water_probabilities["scene_core"],
                    water_probabilities["depth_edge_support"],
                )
            ).clamp(0.0, 1.0)
            layer_veil_smoothness_loss = self._weighted_mean(veil_alpha_grad, smooth_weight)
            layer_exclusivity_loss = (
                water_probabilities["empty_freewater"] * water_probabilities["attached_veil"]
                + water_probabilities["empty_freewater"] * water_probabilities["scene_core"]
                + water_probabilities["attached_veil"] * water_probabilities["scene_core"]
            ).mean()

            non_destructive_guard = (1.0 - strict_destructive_water_probability).clamp(0.0, 1.0)
            scene_fidelity_weight = torch.maximum(
                water_probabilities["scene_rescue"],
                scene_responsibility,
            )
            scene_fidelity_weight = torch.maximum(
                scene_fidelity_weight,
                water_probabilities["water_matte_scene_protection"] * 0.65,
            ) * non_destructive_guard
            haze_fidelity_weight = torch.maximum(
                haze_responsibility,
                water_probabilities["anchored_haze"],
            ) * non_destructive_guard
            underwater_rgb_error = torch.abs(rendered_underwater_image - gt_underwater_image)
            responsibility_scene_fidelity_loss = self._weighted_mean(
                underwater_rgb_error,
                scene_fidelity_weight.repeat(1, 1, 3),
            )
            responsibility_haze_fidelity_loss = self._weighted_mean(
                underwater_rgb_error,
                haze_fidelity_weight.repeat(1, 1, 3),
            )
            responsibility_boundary_source = torch.maximum(
                torch.maximum(freewater_responsibility, haze_responsibility),
                scene_responsibility,
            )
            responsibility_boundary_source = torch.maximum(
                responsibility_boundary_source,
                water_probabilities["anchored_freewater"],
            )
            responsibility_boundary_source = torch.maximum(
                responsibility_boundary_source,
                water_probabilities["water_matte_scene_protection"],
            )
            responsibility_boundary = self._image_gradient_magnitude(responsibility_boundary_source)
            boundary_cut = torch.quantile(
                responsibility_boundary.flatten(),
                0.75,
            ).detach().clamp_min(1e-5)
            responsibility_boundary_weight = torch.sigmoid(
                (responsibility_boundary - boundary_cut)
                / boundary_cut
                * self.config.water_null_edge_sharpness
            )
            responsibility_boundary_weight = responsibility_boundary_weight * non_destructive_guard
            rendered_underwater_gradient = self._image_gradient_magnitude(rendered_underwater_image)
            gt_underwater_gradient = self._image_gradient_magnitude(gt_underwater_image)
            responsibility_boundary_fidelity_loss = self._weighted_mean(
                torch.abs(rendered_underwater_gradient - gt_underwater_gradient),
                responsibility_boundary_weight,
            )

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
            "responsibility_scene_fidelity_loss": responsibility_scene_fidelity_loss * self.config.responsibility_scene_fidelity_loss_weight,
            "responsibility_haze_fidelity_loss": responsibility_haze_fidelity_loss * self.config.responsibility_haze_fidelity_loss_weight,
            "responsibility_boundary_fidelity_loss": responsibility_boundary_fidelity_loss * self.config.responsibility_boundary_fidelity_loss_weight,
            "layer_empty_black_loss": layer_empty_black_loss * self.config.layer_empty_black_loss_weight,
            "layer_attached_chroma_loss": layer_attached_chroma_loss * self.config.layer_attached_chroma_loss_weight,
            "layer_detail_preservation_loss": layer_detail_preservation_loss * self.config.layer_detail_preservation_loss_weight,
            "layer_veil_smoothness_loss": layer_veil_smoothness_loss * self.config.layer_veil_smoothness_loss_weight,
            "layer_exclusivity_loss": layer_exclusivity_loss * self.config.layer_exclusivity_loss_weight,
            "water_calibration_l2_loss": water_calibration_l2_loss * self.config.water_calibrator_l2_weight,
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

    def _get_image_metrics_and_images_nextgen(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Evaluate without conditioning the clean render on test RGB or depth."""
        gt_rgb_hwc = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        predicted_hwc = outputs["rgb"]
        combined_rgb = torch.cat([gt_rgb_hwc, predicted_hwc], dim=1)
        gt_rgb = gt_rgb_hwc.permute(2, 0, 1).unsqueeze(0)
        predicted_rgb = predicted_hwc.permute(2, 0, 1).unsqueeze(0)

        psnr = self.psnr(gt_rgb, predicted_rgb)
        if gt_rgb.shape[-1] > 800 or gt_rgb.shape[-2] > 800:
            ssim = self.split_and_calculate_ssim(self.ssim, gt_rgb, predicted_rgb)
        else:
            ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)
        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim),
            "lpips": float(lpips),
        }

        if self.config.color_corrected_metrics:
            corrected = color_correct(predicted_hwc, gt_rgb_hwc).permute(2, 0, 1).unsqueeze(0)
            cc_psnr = self.psnr(gt_rgb, corrected)
            if gt_rgb.shape[-1] > 800 or gt_rgb.shape[-2] > 800:
                cc_ssim = self.split_and_calculate_ssim(self.ssim, gt_rgb, corrected)
            else:
                cc_ssim = self.ssim(gt_rgb, corrected)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(self.lpips(gt_rgb, corrected))

        gt_depth = batch["depth_image"].to(self.device)
        gt_depth = gt_depth / gt_depth.amax().clamp_min(1e-6)
        depth = outputs["depth"]
        depth_normalized = depth / depth.amax().clamp_min(1e-6)
        far_distance = max(float(self.config.water_transport_far_distance), 1e-6)
        images_dict = {
            "img": combined_rgb,
            "accumulation": outputs["accumulation"],
            "intrinsic_color_render": outputs["intrinsic_color_render"],
            "intrinsic_color_render_raw": outputs["intrinsic_color_render_raw"],
            "intrinsic_visibility": outputs["intrinsic_visibility"],
            "rendered_underwater_image": predicted_hwc,
            "water_background_image": outputs["water_background_image"],
            "background_backscatter_coefficients": outputs[
                "background_backscatter_coefficients"
            ].clamp(0.0, 1.0),
            "background_attenuation_coefficients": outputs[
                "background_attenuation_coefficients"
            ].clamp(0.0, 1.0),
            "direct_transmittance": outputs["direct_transmittance"],
            "optical_depth": (outputs["optical_depth"] / far_distance).clamp(0.0, 1.0),
            "depth_GT": gt_depth,
            "depth_ED": depth_normalized,
        }
        return metrics_dict, images_dict

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
        if self._uses_nextgen_water_model():
            return self._get_image_metrics_and_images_nextgen(outputs, batch)
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        gt_rgb_image = gt_rgb
        gt_depth = batch["depth_image"]
        gt_depth = gt_depth / gt_depth.max()

        intrinsic_raw = outputs.get("intrinsic_color_render_raw", outputs["intrinsic_color_render"])
        eval_water_probabilities = None
        if self.config.enable_water_null_occupancy and not self._uses_surface_carrier():
            eval_water_probabilities = self._build_water_null_probabilities(
                gt_depth.to(self.device),
                gt_rgb_image,
                outputs["water_background_image"],
                intrinsic_raw,
                outputs["accumulation"],
            )
        predicted_rgb = outputs["rgb"]
        predicted_rgb_image = predicted_rgb
        cc_rgb = None

        combined_rgb = torch.cat([gt_rgb, predicted_rgb_image], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb_image, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb_image, -1, 0)[None, ...]

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

        intrinsic_clean = outputs["intrinsic_color_render"]
        images_dict['intrinsic_color_render_raw'] = intrinsic_raw
        images_dict['rendered_underwater_image'] = predicted_rgb_image
         
        images_dict['water_background_image'] = outputs["water_background_image"]   
        images_dict['background_backscatter_coefficients'] = outputs["background_backscatter_coefficients"]   
        images_dict['background_attenuation_coefficients'] = outputs["background_attenuation_coefficients"]
        images_dict['water_calibration_ambient_delta'] = outputs.get(
            "water_calibration_ambient_delta",
            torch.zeros_like(outputs["water_background_image"]),
        )
        images_dict['water_calibration_backscatter_delta'] = outputs.get(
            "water_calibration_backscatter_delta",
            torch.zeros_like(outputs["water_background_image"]),
        )
        images_dict['water_calibration_attenuation_delta'] = outputs.get(
            "water_calibration_attenuation_delta",
            torch.zeros_like(outputs["water_background_image"]),
        )
            
        images_dict['depth_GT'] = gt_depth
        if self.config.enable_water_null_occupancy and not self._uses_surface_carrier():
            if eval_water_probabilities is None:
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
            eval_freewater_responsibility = eval_water_probabilities["responsibility_freewater"]
            eval_haze_responsibility = eval_water_probabilities["responsibility_haze"]
            eval_scene_responsibility = eval_water_probabilities["responsibility_scene"]
            eval_uncertainty = eval_water_probabilities["responsibility_uncertainty"]
            eval_strict_pure_probability = self._build_strict_water_null_probability(
                eval_pure_water_probability,
                self.config.water_null_strict_threshold,
                self.config.water_pure_max_fraction,
            )
            eval_strict_freewater_probability = self._build_strict_water_null_probability(
                eval_water_probabilities["empty_freewater"],
                self.config.layer_empty_freewater_strict_threshold,
                self.config.responsibility_freewater_max_fraction,
            )
            eval_strict_haze_probability = self._build_strict_water_null_probability(
                eval_haze_responsibility,
                self.config.responsibility_strict_haze_threshold,
                self.config.responsibility_haze_max_fraction,
            )
            eval_strict_transition_probability = torch.maximum(
                self._build_strict_water_null_probability(
                    eval_transition_water_probability,
                    self.config.water_transition_strict_threshold,
                    self.config.water_transition_max_fraction,
                ),
                eval_strict_haze_probability,
            )
            eval_strict_water_probability = torch.maximum(
                torch.maximum(eval_strict_pure_probability, eval_strict_freewater_probability),
                eval_strict_transition_probability,
            )
            eval_alpha_water = torch.maximum(
                outputs.get("water_accumulation", torch.zeros_like(outputs["accumulation"])),
                outputs["accumulation"] * eval_strict_freewater_probability,
            )
            eval_intrinsic_visibility = self._build_intrinsic_visibility_gate(
                outputs["accumulation"],
                eval_water_probability,
                eval_alpha_water,
                eval_water_probabilities["scene_rescue"],
                eval_water_probabilities["residual_water"],
                eval_water_probabilities["connected_freewater"],
                eval_freewater_responsibility,
                eval_haze_responsibility,
                eval_scene_responsibility,
                eval_uncertainty,
            )
            layered_input = intrinsic_clean if self._uses_surface_carrier() else intrinsic_raw
            layered_clean_outputs = self._compose_layered_clean_render(
                layered_input,
                outputs["water_background_image"],
                eval_water_probabilities,
            )
            intrinsic_clean = layered_clean_outputs["clean"]
            images_dict['intrinsic_visibility'] = eval_intrinsic_visibility
            images_dict['layered_deveiled_intrinsic_color_render'] = layered_clean_outputs["deveiled"]
            images_dict['layered_veil_blend'] = layered_clean_outputs["veil_blend"]
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
            images_dict['anchored_freewater_probability'] = eval_water_probabilities["anchored_freewater"]
            images_dict['anchored_haze_probability'] = eval_water_probabilities["anchored_haze"]
            images_dict['water_anchor_similarity'] = eval_water_probabilities["water_anchor_similarity"]
            images_dict['water_anchor_input_chroma'] = eval_water_probabilities["water_anchor_input_chroma"]
            images_dict['water_matte_scene_protection'] = eval_water_probabilities["water_matte_scene_protection"]
            images_dict['water_matte_low_structure'] = eval_water_probabilities["water_matte_low_structure"]
            images_dict['empty_freewater_probability'] = eval_water_probabilities["empty_freewater"]
            images_dict['attached_veil_probability'] = eval_water_probabilities["attached_veil"]
            images_dict['scene_core_probability'] = eval_water_probabilities["scene_core"]
            images_dict['uncertain_boundary_probability'] = eval_water_probabilities["uncertain_boundary"]
            images_dict['water_color_context'] = eval_water_probabilities["water_color_context"]
            images_dict['veil_alpha'] = eval_water_probabilities["veil_alpha"]
            images_dict['responsibility_freewater'] = eval_freewater_responsibility
            images_dict['responsibility_haze'] = eval_haze_responsibility
            images_dict['responsibility_scene'] = eval_scene_responsibility
            images_dict['responsibility_uncertainty'] = eval_uncertainty

        images_dict['intrinsic_color_render'] = intrinsic_clean
        
        depth_ED = outputs["depth"]
        q99 = torch.quantile(depth_ED.flatten(), 0.99)
        depth_ED_clipped = torch.clamp(depth_ED, max=q99)
        depth_ED_normalized = depth_ED_clipped / q99
        images_dict['depth_ED'] = depth_ED_normalized

            
        return metrics_dict, images_dict
