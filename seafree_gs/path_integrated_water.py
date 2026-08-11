"""Compact 3D water field with camera-frustum prefix integration."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


TransportPrefix = Dict[str, Union[torch.Tensor, int]]


class CompactHeterogeneousWaterField(nn.Module):
    """A bounded physical macro-grid around the directional WPP anchor.

    The four channels control density, spectral water type, the relative
    backscatter ratio, and ambient intensity.  They cannot emit arbitrary RGB
    texture.  Removing the spatial DC component structurally assigns global
    water appearance to WPP and local variation to this field.
    """

    def __init__(
        self,
        resolution: int = 16,
        ambient_residual_scale: float = 0.04,
        coefficient_log_scale: float = 0.20,
    ) -> None:
        super().__init__()
        self.resolution = max(int(resolution), 4)
        self.ambient_residual_scale = max(float(ambient_residual_scale), 0.0)
        self.coefficient_log_scale = max(float(coefficient_log_scale), 0.0)
        self.volume = nn.Parameter(
            torch.zeros(1, 4, self.resolution, self.resolution, self.resolution)
        )
        self.register_buffer(
            "free_space_support",
            torch.zeros(1, 1, self.resolution, self.resolution, self.resolution),
        )
        self.register_buffer(
            "medium_optical_mass",
            torch.zeros(1, 1, self.resolution, self.resolution, self.resolution),
        )
        self.register_buffer(
            "surface_endpoint_mass",
            torch.zeros(1, 1, self.resolution, self.resolution, self.resolution),
        )
        self.register_buffer("support_observation_mass", torch.zeros((), dtype=torch.float32))
        self.register_buffer("latent_running_mean", torch.zeros(4, dtype=torch.float32))
        self.register_buffer("latent_observation_mass", torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "ambient_anchor", torch.tensor([0.18, 0.38, 0.48], dtype=torch.float32)
        )
        self.register_buffer("ambient_observation_mass", torch.zeros((), dtype=torch.float32))
        self.register_buffer("aabb_min", torch.full((3,), -1.25, dtype=torch.float32))
        self.register_buffer("aabb_max", torch.full((3,), 1.25, dtype=torch.float32))
        self.register_buffer("scene_scale", torch.tensor(2.5, dtype=torch.float32))

    @property
    def planes(self) -> torch.Tensor:
        """Compatibility view used by old diagnostics."""
        return self.volume

    @torch.no_grad()
    def set_bounds(self, points: torch.Tensor) -> None:
        if points.numel() == 0:
            return
        values = points.detach().float().reshape(-1, 3)
        if values.shape[0] >= 64:
            lower = torch.quantile(values, 0.01, dim=0)
            upper = torch.quantile(values, 0.99, dim=0)
        else:
            lower = values.amin(dim=0)
            upper = values.amax(dim=0)
        extent = (upper - lower).clamp_min(1e-3)
        # Camera centres in normalized Nerfstudio scenes normally sit just
        # outside SceneBox. This margin keeps the whole useful water path valid.
        padding = 0.25 * extent
        bounded_min = lower - padding
        bounded_max = upper + padding
        self.aabb_min.copy_(bounded_min.to(self.aabb_min))
        self.aabb_max.copy_(bounded_max.to(self.aabb_max))
        self.scene_scale.copy_(
            torch.linalg.vector_norm(bounded_max - bounded_min)
            .clamp_min(1e-3)
            .to(self.scene_scale)
        )

    def normalize_distance(
        self, distances: torch.Tensor, depth_scale: Optional[float] = 10.0
    ) -> torch.Tensor:
        if depth_scale is None:
            return distances / self.scene_scale.to(distances).clamp_min(1e-4)
        return distances / max(float(depth_scale), 1e-4)

    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        lower = self.aabb_min.to(positions)
        extent = (self.aabb_max - self.aabb_min).to(positions).clamp_min(1e-4)
        return (positions - lower) / extent * 2.0 - 1.0

    def ray_box_interval(
        self, origins: torch.Tensor, directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if origins.dim() == 1:
            origins = origins.unsqueeze(0).expand_as(directions)
        lower = self.aabb_min.to(directions)
        upper = self.aabb_max.to(directions)
        safe_direction = torch.where(
            directions.abs() < 1e-6,
            torch.where(directions >= 0.0, directions.new_full((), 1e-6), directions.new_full((), -1e-6)),
            directions,
        )
        t0 = (lower - origins) / safe_direction
        t1 = (upper - origins) / safe_direction
        entry = torch.minimum(t0, t1).amax(dim=-1).clamp_min(0.0)
        exit = torch.maximum(t0, t1).amin(dim=-1)
        valid = exit > entry + 1e-5
        exit = torch.where(valid, exit, entry + 1e-4)
        return entry, exit, valid

    def _centered_volume(self) -> torch.Tensor:
        # The learned residual is structurally low frequency.  This removes the
        # high-frequency capacity that otherwise competes with Gaussian texture.
        smooth = F.avg_pool3d(
            F.pad(self.volume.float(), (1, 1, 1, 1, 1, 1), mode="replicate"),
            kernel_size=3,
            stride=1,
        )
        return smooth - smooth.mean(dim=(2, 3, 4), keepdim=True)

    def _sample_grid(self, source: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.numel() == 0:
            return positions.new_zeros((*positions.shape[:-1], source.shape[1]))
        original_shape = positions.shape[:-1]
        normalized = self.normalize_positions(positions.reshape(-1, 3)).float()
        grid = normalized.reshape(1, 1, 1, -1, 3)
        sampled = F.grid_sample(
            source.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        raw = sampled[0, :, 0, 0].transpose(0, 1)
        return raw.reshape(*original_shape, source.shape[1]).to(positions)

    def _sample_volume(self, positions: torch.Tensor) -> torch.Tensor:
        return self._sample_grid(self._centered_volume(), positions)

    def _sample_free_space_support(self, positions: torch.Tensor) -> torch.Tensor:
        support = F.avg_pool3d(
            F.pad(
                self.free_space_support.float(),
                (1, 1, 1, 1, 1, 1),
                mode="replicate",
            ),
            kernel_size=3,
            stride=1,
        )
        return self._sample_grid(support, positions).clamp(0.0, 1.0)

    @torch.no_grad()
    def update_free_space_support(
        self,
        positions: torch.Tensor,
        medium_probability: torch.Tensor,
        surface_probability: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        opacity: Optional[torch.Tensor] = None,
        camera_origin: Optional[torch.Tensor] = None,
        ema: float = 0.85,
        chunk_size: int = 262144,
    ) -> None:
        """Accumulate trusted camera-to-carrier water segments on the model device."""
        if positions.numel() == 0:
            return
        points = positions.detach().reshape(-1, 3)
        count = points.shape[0]
        medium = medium_probability.detach().float().reshape(-1)[:count].clamp(0.0, 1.0)
        surface = surface_probability.detach().float().reshape(-1)[:count].clamp(0.0, 1.0)
        if medium.shape[0] != count or surface.shape[0] != count:
            return
        if confidence is None:
            reliability = torch.ones(count, device=points.device, dtype=torch.float32)
        else:
            if confidence.numel() < count:
                return
            reliability = confidence.detach().float().reshape(-1)[:count].clamp(0.0, 1.0)
        optical_thickness = torch.ones_like(reliability)
        if opacity is not None:
            if opacity.numel() < count:
                return
            alpha = opacity.detach().float().reshape(-1)[:count].clamp(0.0, 1.0 - 1e-6)
            optical_thickness = (-torch.log1p(-alpha)).clamp(0.05, 4.0)
            reliability = reliability * optical_thickness

        total_voxels = self.resolution**3
        medium_mass = torch.zeros(total_voxels, device=points.device, dtype=torch.float32)
        surface_mass = torch.zeros_like(medium_mass)
        observed_mass = torch.zeros((), device=points.device, dtype=torch.float32)
        lower = self.aabb_min.to(points)
        extent = (self.aabb_max - self.aabb_min).to(points).clamp_min(1e-4)
        chunk_size = max(int(chunk_size), 16384)
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            chunk_points = points[start:end]
            unit = (chunk_points - lower) / extent
            inside = ((unit >= 0.0) & (unit <= 1.0)).all(dim=-1)
            voxel = (unit.clamp(0.0, 1.0) * (self.resolution - 1)).round().to(torch.int64)
            flat_index = (
                voxel[:, 2] * self.resolution * self.resolution
                + voxel[:, 1] * self.resolution
                + voxel[:, 0]
            )
            valid_weight = reliability[start:end] * inside.to(reliability)
            medium_mass.scatter_add_(0, flat_index, valid_weight * medium[start:end])
            surface_mass.scatter_add_(0, flat_index, valid_weight * surface[start:end])

            if camera_origin is not None:
                origin = camera_origin.detach().float().reshape(1, 3).to(chunk_points)
                fractions = chunk_points.new_tensor([0.15, 0.35, 0.55, 0.75, 0.90])
                segment_points = origin[:, None, :] + fractions[None, :, None] * (
                    chunk_points[:, None, :] - origin[:, None, :]
                )
                segment_unit = (segment_points - lower) / extent
                segment_inside = ((segment_unit >= 0.0) & (segment_unit <= 1.0)).all(dim=-1)
                segment_voxel = (
                    segment_unit.clamp(0.0, 1.0) * (self.resolution - 1)
                ).round().to(torch.int64)
                segment_index = (
                    segment_voxel[..., 2] * self.resolution * self.resolution
                    + segment_voxel[..., 1] * self.resolution
                    + segment_voxel[..., 0]
                )
                ray_trust = valid_weight * (surface[start:end] + medium[start:end]).clamp(0.0, 1.0)
                segment_weight = (
                    ray_trust[:, None] * segment_inside.to(ray_trust) / fractions.numel()
                )
                medium_mass.scatter_add_(
                    0, segment_index.reshape(-1), segment_weight.reshape(-1)
                )
                observed_mass.add_(segment_weight.sum())
            observed_mass.add_(valid_weight.sum())

        shape = (1, 1, self.resolution, self.resolution, self.resolution)
        keep = min(max(float(ema), 0.0), 0.9999)
        self.medium_optical_mass.mul_(keep).add_(medium_mass.reshape(shape), alpha=1.0 - keep)
        self.surface_endpoint_mass.mul_(keep).add_(surface_mass.reshape(shape), alpha=1.0 - keep)
        medium_context = F.avg_pool3d(self.medium_optical_mass.float(), 3, 1, 1)
        surface_exclusion = F.max_pool3d(self.surface_endpoint_mass.float(), 3, 1, 1)
        total_context = medium_context + surface_exclusion
        target = medium_context / total_context.clamp_min(1e-5)
        target = (target * (1.0 - 0.90 * surface_exclusion / total_context.clamp_min(1e-5))).clamp(0.0, 1.0)
        known_context = total_context > 1e-5
        updated_support = torch.where(known_context, target, self.free_space_support.float())
        self.free_space_support.lerp_(updated_support.to(self.free_space_support), 1.0 - keep)
        self.support_observation_mass.mul_(keep).add_(
            observed_mass.to(self.support_observation_mass) * (1.0 - keep)
        )

    def residual_raw(
        self, positions: torch.Tensor, directions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del directions
        return self._sample_volume(positions)

    @torch.no_grad()
    def observe_latents(
        self, raw_latents: torch.Tensor, weights: torch.Tensor, ema: float = 0.98
    ) -> None:
        # Diagnostics only: DC removal is structural and does not depend on a
        # classifier, so a failed carrier can no longer disable field centring.
        if raw_latents.numel() == 0:
            return
        raw = raw_latents.detach().float().reshape(-1, 4)
        weight = weights.detach().float().reshape(-1, 1).clamp(0.0, 1.0)
        if raw.shape[0] != weight.shape[0]:
            return
        denominator = weight.sum()
        observed = (raw * weight).sum(dim=0) / denominator.clamp_min(1e-5)
        valid = denominator > 1e-3
        keep = min(max(float(ema), 0.0), 0.9999)
        self.latent_running_mean.lerp_(
            observed.to(self.latent_running_mean), valid.to(self.latent_running_mean) * (1.0 - keep)
        )
        next_mass = self.latent_observation_mass * keep + denominator.to(
            self.latent_observation_mass
        ) * (1.0 - keep)
        self.latent_observation_mass.copy_(
            torch.where(valid, next_mass, self.latent_observation_mass)
        )

    def multipliers(
        self,
        positions: torch.Tensor,
        spatial_gate: Optional[torch.Tensor] = None,
        ambient_scale: Optional[float] = None,
        coefficient_log_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self._sample_volume(positions)
        gate = self._sample_free_space_support(positions)
        if spatial_gate is not None:
            external_gate = spatial_gate.to(raw)
            while external_gate.dim() < raw.dim():
                external_gate = external_gate.unsqueeze(-1)
            gate = gate * external_gate.clamp(0.0, 1.0)
        raw = raw * gate
        bounded = torch.tanh(raw)
        ambient_scale = self.ambient_residual_scale if ambient_scale is None else max(float(ambient_scale), 0.0)
        coefficient_log_scale = (
            self.coefficient_log_scale
            if coefficient_log_scale is None
            else max(float(coefficient_log_scale), 0.0)
        )
        density = torch.exp(coefficient_log_scale * bounded[..., 0:1])
        water_type = coefficient_log_scale * bounded[..., 1:2]
        ratio = torch.exp(coefficient_log_scale * bounded[..., 2:3])
        ambient = 1.0 + ambient_scale * bounded[..., 3:4]
        attenuation_slope = raw.new_tensor([0.65, 0.0, -0.65])
        backscatter_slope = raw.new_tensor([-0.35, 0.0, 0.35])
        attenuation_multiplier = density * torch.exp(water_type * attenuation_slope)
        backscatter_multiplier = density * ratio * torch.exp(water_type * backscatter_slope)
        ambient_multiplier = ambient.expand_as(attenuation_multiplier)
        return ambient_multiplier, backscatter_multiplier, attenuation_multiplier, raw

    def apply_to_base(
        self,
        ambient: torch.Tensor,
        backscatter: torch.Tensor,
        attenuation: torch.Tensor,
        positions: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
        ambient_scale: Optional[float] = None,
        coefficient_log_scale: Optional[float] = None,
        spatial_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del directions
        _, backscatter_mul, attenuation_mul, raw = self.multipliers(
            positions,
            spatial_gate=spatial_gate,
            ambient_scale=ambient_scale,
            coefficient_log_scale=coefficient_log_scale,
        )
        corrected_ambient = self.correct_ambient(
            ambient, raw, ambient_scale=ambient_scale
        )
        return (
            corrected_ambient.clamp(0.0, 1.0),
            backscatter * backscatter_mul,
            attenuation * attenuation_mul,
            raw,
        )

    def correct_ambient(
        self,
        ambient: torch.Tensor,
        raw: torch.Tensor,
        ambient_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply a bounded residual toward the observed free-water light anchor."""
        resolved_scale = (
            self.ambient_residual_scale
            if ambient_scale is None
            else max(float(ambient_scale), 0.0)
        )
        strength = resolved_scale * torch.tanh(raw[..., 3:4])
        anchor = self.ambient_anchor.to(ambient)
        return ambient + strength * (anchor - ambient)

    def properties_for_directions(
        self,
        directions: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if positions is None:
            positions = torch.zeros_like(directions)
        count = directions.shape[0]
        ambient = self.ambient_anchor.to(directions).expand(count, -1)
        backscatter = directions.new_tensor([0.08, 0.16, 0.22]).expand(count, -1)
        attenuation = directions.new_tensor([0.30, 0.16, 0.08]).expand(count, -1)
        ambient, backscatter, attenuation, _ = self.apply_to_base(
            ambient, backscatter, attenuation, positions
        )
        return ambient, backscatter, attenuation

    def query_spatial(self, positions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        directions = torch.zeros_like(positions)
        directions[..., 2] = 1.0
        ambient, backscatter, attenuation = self.properties_for_directions(
            directions, positions
        )
        return torch.logit(ambient.clamp(1e-5, 1.0 - 1e-5)), backscatter, attenuation

    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.properties_for_directions(directions, positions)

    @torch.no_grad()
    def observe_freewater(
        self,
        image: torch.Tensor,
        responsibility: torch.Tensor,
        ema: float = 0.97,
    ) -> None:
        rgb = image[..., :3].detach().float().clamp(0.0, 1.0)
        weight = responsibility[..., :1].detach().float().clamp(0.0, 1.0)
        decisive = (weight > 0.70).to(weight) * weight
        denominator = decisive.sum()
        observed = (rgb * decisive).sum(dim=(0, 1)) / denominator.clamp_min(1e-5)
        valid = denominator > 16.0
        keep = min(max(float(ema), 0.0), 0.9999)
        self.ambient_anchor.lerp_(
            observed.to(self.ambient_anchor), valid.to(self.ambient_anchor) * (1.0 - keep)
        )
        updated_mass = self.ambient_observation_mass * keep + denominator.to(
            self.ambient_observation_mass
        ) * (1.0 - keep)
        self.ambient_observation_mass.copy_(
            torch.where(valid, updated_mass, self.ambient_observation_mass)
        )


class PathIntegratedWaterRenderer(nn.Module):
    """Build and query a compact front-to-back transport prefix per camera."""

    def __init__(
        self,
        field: CompactHeterogeneousWaterField,
        query_downscale: int = 8,
        depth_scale: float = 10.0,
        far_distance: float = 4.0,
        num_depth_bins: int = 8,
        query_chunk_size: int = 262144,
    ) -> None:
        super().__init__()
        self.field = field
        self.query_downscale = max(int(query_downscale), 1)
        self.depth_scale = max(float(depth_scale), 1e-4)
        self.far_distance = max(float(far_distance), 1e-3)
        self.num_depth_bins = max(int(num_depth_bins), 4)
        self.query_chunk_size = max(int(query_chunk_size), 16384)

    @staticmethod
    def _upsample(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
        low = value.permute(2, 0, 1).unsqueeze(0)
        return F.interpolate(
            low, size=(height, width), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0)

    def _low_resolution_rays(
        self,
        height: int,
        width: int,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, int, int]:
        low_h = max(1, (height + self.query_downscale - 1) // self.query_downscale)
        low_w = max(1, (width + self.query_downscale - 1) // self.query_downscale)
        ys = (torch.arange(low_h, device=intrinsics.device, dtype=dtype) + 0.5) * height / low_h
        xs = (torch.arange(low_w, device=intrinsics.device, dtype=dtype) + 0.5) * width / low_w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        fx, fy = intrinsics[0, 0, 0], intrinsics[0, 1, 1]
        cx, cy = intrinsics[0, 0, 2], intrinsics[0, 1, 2]
        camera_dirs = torch.stack(
            [(xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx)], dim=-1
        )
        camera_dirs = F.normalize(camera_dirs, dim=-1)
        world_dirs = camera_dirs @ camera_to_world[0, :3, :3].transpose(0, 1)
        return F.normalize(world_dirs, dim=-1), low_h, low_w

    def build_frustum_prefix(
        self,
        height: int,
        width: int,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        dtype: torch.dtype,
        query_base_properties: Callable[[torch.Tensor], Tuple[torch.Tensor, ...]],
        enabled: bool,
        base_property_images: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
    ) -> TransportPrefix:
        directions, low_h, low_w = self._low_resolution_rays(
            height, width, intrinsics, camera_to_world, dtype
        )
        flat_directions = directions.reshape(-1, 3)
        origin = camera_to_world[0, :3, 3].detach().to(flat_directions)
        origins = origin.unsqueeze(0).expand_as(flat_directions)
        entry, exit, valid = self.field.ray_box_interval(origins, flat_directions)
        fallback_exit = flat_directions.new_full(
            exit.shape, self.depth_scale * self.far_distance
        )
        entry = torch.where(valid, entry, torch.zeros_like(entry))
        exit = torch.where(valid, exit, fallback_exit)
        span = (exit - entry).clamp_min(1e-4)

        endpoints = torch.linspace(
            0.0, 1.0, self.num_depth_bins + 1, device=flat_directions.device, dtype=dtype
        )
        midpoint_fraction = 0.5 * (endpoints[:-1] + endpoints[1:])
        midpoint_distance = entry.unsqueeze(0) + midpoint_fraction[:, None] * span.unsqueeze(0)
        segment_distance = span.unsqueeze(0) / self.num_depth_bins
        positions = origins.unsqueeze(0) + midpoint_distance[..., None] * flat_directions.unsqueeze(0)

        if base_property_images is None:
            ambient, backscatter, attenuation, *_ = query_base_properties(flat_directions)
            ambient = ambient.reshape(1, -1, 3)
            backscatter = backscatter.reshape(1, -1, 3)
            attenuation = attenuation.reshape(1, -1, 3)
        else:
            property_image = torch.cat(base_property_images, dim=-1)
            low_properties = F.interpolate(
                property_image.permute(2, 0, 1).unsqueeze(0),
                size=(low_h, low_w),
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0).reshape(-1, 9)
            ambient, backscatter, attenuation = low_properties.split(3, dim=-1)
            ambient = ambient.unsqueeze(0)
            backscatter = backscatter.unsqueeze(0)
            attenuation = attenuation.unsqueeze(0)
        if enabled:
            _, backscatter_mul, attenuation_mul, raw = self.field.multipliers(positions)
            valid_gate = valid.reshape(1, -1, 1)
            backscatter_mul = torch.where(
                valid_gate, backscatter_mul, torch.ones_like(backscatter_mul)
            )
            attenuation_mul = torch.where(
                valid_gate, attenuation_mul, torch.ones_like(attenuation_mul)
            )
            raw = torch.where(valid_gate, raw, torch.zeros_like(raw))
        else:
            backscatter_mul = positions.new_ones((*positions.shape[:-1], 3))
            attenuation_mul = positions.new_ones((*positions.shape[:-1], 3))
            raw = positions.new_zeros((*positions.shape[:-1], 4))

        heterogeneous_ambient = self.field.correct_ambient(ambient, raw)

        normalized_segment = segment_distance[..., None] / self.depth_scale
        direct_segment = torch.exp(-attenuation * attenuation_mul * normalized_segment)
        scatter_transmittance_segment = torch.exp(
            -backscatter * backscatter_mul * normalized_segment
        )
        heterogeneous_direct_end = torch.cumprod(direct_segment, dim=0)
        homogeneous_direct_segment = torch.exp(
            -attenuation * normalized_segment
        ).expand_as(direct_segment)
        homogeneous_direct_end = torch.cumprod(homogeneous_direct_segment, dim=0)
        scatter_transmittance_before = torch.cat(
            [
                torch.ones_like(scatter_transmittance_segment[:1]),
                torch.cumprod(scatter_transmittance_segment[:-1], dim=0),
            ],
            dim=0,
        )
        scatter_source = (
            heterogeneous_ambient
            * (1.0 - scatter_transmittance_segment)
            * scatter_transmittance_before
        )
        heterogeneous_scatter_end = torch.cumsum(scatter_source, dim=0)
        homogeneous_scatter_segment = torch.exp(
            -backscatter * normalized_segment
        ).expand_as(scatter_transmittance_segment)
        homogeneous_scatter_transmittance_before = torch.cat(
            [
                torch.ones_like(homogeneous_scatter_segment[:1]),
                torch.cumprod(homogeneous_scatter_segment[:-1], dim=0),
            ],
            dim=0,
        )
        homogeneous_scatter_end = torch.cumsum(
            ambient
            * (1.0 - homogeneous_scatter_segment)
            * homogeneous_scatter_transmittance_before,
            dim=0,
        )
        heterogeneous_scatter_transmittance_end = torch.cumprod(
            scatter_transmittance_segment, dim=0
        )
        homogeneous_scatter_transmittance_end = torch.cumprod(
            homogeneous_scatter_segment, dim=0
        )
        base_length = (
            entry[None, :, None] + endpoints[:, None, None] * span[None, :, None]
        ) / self.depth_scale
        base_direct_prefix = torch.exp(-attenuation * base_length)
        base_scatter_prefix = ambient * (1.0 - torch.exp(-backscatter * base_length))
        pre_direct = torch.exp(-attenuation * entry[None, :, None] / self.depth_scale)
        pre_scatter_transmittance = torch.exp(
            -backscatter * entry[None, :, None] / self.depth_scale
        )
        direct_residual = torch.cat(
            [
                torch.zeros_like(heterogeneous_direct_end[:1]),
                pre_direct * (heterogeneous_direct_end - homogeneous_direct_end),
            ],
            dim=0,
        )
        scatter_residual = torch.cat(
            [
                torch.zeros_like(heterogeneous_scatter_end[:1]),
                pre_scatter_transmittance
                * (heterogeneous_scatter_end - homogeneous_scatter_end),
            ],
            dim=0,
        )
        scatter_transmittance_residual = torch.cat(
            [
                torch.zeros_like(heterogeneous_scatter_transmittance_end[:1]),
                pre_scatter_transmittance
                * (
                    heterogeneous_scatter_transmittance_end
                    - homogeneous_scatter_transmittance_end
                ),
            ],
            dim=0,
        )
        direct_prefix = base_direct_prefix + direct_residual
        scatter_prefix = base_scatter_prefix + scatter_residual
        shape = (self.num_depth_bins + 1, low_h, low_w, 3)
        return {
            "direct": direct_prefix.reshape(shape),
            "scatter": scatter_prefix.reshape(shape),
            "direct_residual": direct_residual.reshape(shape),
            "scatter_residual": scatter_residual.reshape(shape),
            "scatter_transmittance_residual": scatter_transmittance_residual.reshape(shape),
            "base_scatter": base_scatter_prefix.reshape(shape),
            "entry": entry.reshape(low_h, low_w),
            "exit": exit.reshape(low_h, low_w),
            "directions": directions,
            "ambient": ambient.reshape(low_h, low_w, 3),
            "backscatter": backscatter.reshape(low_h, low_w, 3),
            "attenuation": attenuation.reshape(low_h, low_w, 3),
            "mean_backscatter_multiplier": backscatter_mul.mean(dim=0).reshape(low_h, low_w, 3),
            "mean_attenuation_multiplier": attenuation_mul.mean(dim=0).reshape(low_h, low_w, 3),
            "camera_origin": origin,
            # Keep image dimensions as host metadata. Reading a CUDA scalar here
            # would otherwise synchronize every training render.
            "height": height,
            "width": width,
            "intrinsics": intrinsics[0],
        }

    @staticmethod
    def _sample_volume_at(
        volume: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        source = volume.permute(3, 0, 1, 2).unsqueeze(0)
        grid = coordinates.reshape(1, 1, 1, -1, 3)
        sampled = F.grid_sample(
            source.float(), grid.float(), mode="bilinear", padding_mode="border", align_corners=True
        )
        return sampled[0, :, 0, 0].transpose(0, 1).to(coordinates)

    def sample_projected_paths(
        self,
        prefix: TransportPrefix,
        means2d: torch.Tensor,
        depths: torch.Tensor,
        visible_ids: torch.Tensor,
        base_ambient: Optional[torch.Tensor] = None,
        base_backscatter: Optional[torch.Tensor] = None,
        base_attenuation: Optional[torch.Tensor] = None,
        ray_directions: Optional[torch.Tensor] = None,
        path_distances: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if visible_ids.numel() == 0:
            empty = means2d.new_zeros((0, 3))
            return {"direct_transmittance": empty, "backscatter_radiance": empty}
        xy = means2d[visible_ids]
        intrinsics = prefix["intrinsics"]
        assert isinstance(intrinsics, torch.Tensor)
        intrinsics = intrinsics.to(xy)
        if path_distances is None:
            z = depths[visible_ids].clamp_min(1e-5)
            dx = (xy[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0]
            dy = (xy[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1]
            distance = z * torch.sqrt(dx * dx + dy * dy + 1.0)
        else:
            distance = path_distances.to(xy).clamp_min(1e-5)
        low_directions = prefix["directions"]
        assert isinstance(low_directions, torch.Tensor)
        height = int(prefix["height"])
        width = int(prefix["width"])
        low_h, low_w = low_directions.shape[:2]
        x_index = xy[:, 0] * low_w / max(width, 1) - 0.5
        y_index = xy[:, 1] * low_h / max(height, 1) - 0.5
        x_coord = 2.0 * x_index / max(low_w - 1, 1) - 1.0
        y_coord = 2.0 * y_index / max(low_h - 1, 1) - 1.0
        grid_2d = torch.stack([x_coord, y_coord], dim=-1).reshape(1, 1, -1, 2)
        exact_interval = ray_directions is not None and path_distances is not None
        if exact_interval:
            camera_origin = prefix["camera_origin"]
            assert isinstance(camera_origin, torch.Tensor)
            entry, exit, valid_path = self.field.ray_box_interval(
                camera_origin.to(xy), ray_directions.to(xy)
            )
        else:
            entry_map, exit_map = prefix["entry"], prefix["exit"]
            assert isinstance(entry_map, torch.Tensor) and isinstance(exit_map, torch.Tensor)
            interval_source = torch.stack([entry_map, exit_map], dim=0).unsqueeze(0)
            interval = F.grid_sample(
                interval_source.float(),
                grid_2d.float(),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[0, :, 0].transpose(0, 1).to(xy)
            entry, exit = interval[:, 0], interval[:, 1]
            valid_path = torch.ones_like(entry, dtype=torch.bool)
        fraction = ((distance - entry) / (exit - entry).clamp_min(1e-4)).clamp(0.0, 1.0)
        coordinates = torch.stack([x_coord, y_coord, fraction * 2.0 - 1.0], dim=-1)

        direct_residual_chunks = []
        scatter_residual_chunks = []
        scatter_transmittance_residual_chunks = []
        direct_residual_prefix = prefix["direct_residual"]
        scatter_residual_prefix = prefix["scatter_residual"]
        scatter_transmittance_residual_prefix = prefix[
            "scatter_transmittance_residual"
        ]
        assert isinstance(direct_residual_prefix, torch.Tensor)
        assert isinstance(scatter_residual_prefix, torch.Tensor)
        assert isinstance(scatter_transmittance_residual_prefix, torch.Tensor)
        for start in range(0, coordinates.shape[0], self.query_chunk_size):
            end = min(start + self.query_chunk_size, coordinates.shape[0])
            direct_residual_chunks.append(
                self._sample_volume_at(direct_residual_prefix, coordinates[start:end])
            )
            scatter_residual_chunks.append(
                self._sample_volume_at(scatter_residual_prefix, coordinates[start:end])
            )
            scatter_transmittance_residual_chunks.append(
                self._sample_volume_at(
                    scatter_transmittance_residual_prefix, coordinates[start:end]
                )
            )
        direct_residual = torch.cat(direct_residual_chunks, dim=0)
        scatter_residual = torch.cat(scatter_residual_chunks, dim=0)
        scatter_transmittance_residual = torch.cat(
            scatter_transmittance_residual_chunks, dim=0
        )
        path_gate = valid_path[:, None].to(direct_residual)
        direct_residual = direct_residual * path_gate
        scatter_residual = scatter_residual * path_gate
        scatter_transmittance_residual = scatter_transmittance_residual * path_gate

        if base_ambient is None or base_backscatter is None or base_attenuation is None:
            ambient_map, backscatter_map, attenuation_map = (
                prefix["ambient"],
                prefix["backscatter"],
                prefix["attenuation"],
            )
            assert isinstance(ambient_map, torch.Tensor)
            assert isinstance(backscatter_map, torch.Tensor)
            assert isinstance(attenuation_map, torch.Tensor)
            property_source = torch.cat(
                [ambient_map, backscatter_map, attenuation_map], dim=-1
            ).permute(2, 0, 1).unsqueeze(0)
            sampled_properties = F.grid_sample(
                property_source.float(),
                grid_2d.float(),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[0, :, 0].transpose(0, 1).to(xy)
            base_ambient, base_backscatter, base_attenuation = sampled_properties.split(3, dim=-1)
        else:
            base_ambient = base_ambient.to(xy)
            base_backscatter = base_backscatter.to(xy)
            base_attenuation = base_attenuation.to(xy)
        normalized_distance = (distance / self.depth_scale).clamp(0.0, self.far_distance)[:, None]
        post_field_distance = (distance - exit).clamp_min(0.0)
        normalized_post_field_distance = (
            post_field_distance / self.depth_scale
        ).clamp(0.0, self.far_distance)[:, None]
        post_direct = torch.exp(-base_attenuation * normalized_post_field_distance)
        post_scatter_source = base_ambient * (
            1.0 - torch.exp(-base_backscatter * normalized_post_field_distance)
        )
        direct_residual = direct_residual * post_direct
        scatter_residual = (
            scatter_residual
            + scatter_transmittance_residual * post_scatter_source
        )
        baseline_direct = torch.exp(-base_attenuation * normalized_distance)
        baseline_scatter = base_ambient * (
            1.0 - torch.exp(-base_backscatter * normalized_distance)
        )
        return {
            "direct_transmittance": baseline_direct + direct_residual,
            "backscatter_radiance": baseline_scatter + scatter_residual,
        }

    def integrate_base_paths(
        self,
        ambient: torch.Tensor,
        backscatter: torch.Tensor,
        attenuation: torch.Tensor,
        origins: torch.Tensor,
        directions: torch.Tensor,
        distances: torch.Tensor,
        spatial_gate: Optional[torch.Tensor] = None,
        observe_medium: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if distances.numel() == 0:
            empty = ambient.new_zeros(ambient.shape)
            return {
                "direct_transmittance": torch.ones_like(empty),
                "backscatter_radiance": empty,
                "ambient": ambient,
                "backscatter": backscatter,
                "attenuation": attenuation,
                "optical_depth": empty,
                "field_latent": ambient.new_zeros((*ambient.shape[:-1], 4)),
            }
        if origins.dim() == 1:
            origins = origins.unsqueeze(0).expand_as(directions)
        clamped_distances = distances.clamp(
            min=0.0, max=self.depth_scale * self.far_distance
        )
        fractions = (
            torch.arange(self.num_depth_bins, device=distances.device, dtype=distances.dtype) + 0.5
        ) / self.num_depth_bins
        positions = origins.unsqueeze(0) + (
            fractions[:, None, None]
            * clamped_distances[None, :, None]
            * directions.unsqueeze(0)
        )
        _, backscatter_mul, attenuation_mul, raw = self.field.multipliers(
            positions, spatial_gate=spatial_gate
        )
        heterogeneous_ambient = self.field.correct_ambient(ambient.unsqueeze(0), raw)
        segment = clamped_distances[None, :, None] / (
            self.depth_scale * self.num_depth_bins
        )
        direct_segment = torch.exp(-attenuation.unsqueeze(0) * attenuation_mul * segment)
        heterogeneous_direct = direct_segment.prod(dim=0)
        homogeneous_direct_segment = torch.exp(
            -attenuation.unsqueeze(0) * segment
        ).expand_as(attenuation_mul)
        homogeneous_direct = homogeneous_direct_segment.prod(dim=0)
        normalized_distance = clamped_distances[:, None] / self.depth_scale
        base_direct = torch.exp(-attenuation * normalized_distance)
        direct = base_direct + (heterogeneous_direct - homogeneous_direct)
        scatter_segment_t = torch.exp(-backscatter.unsqueeze(0) * backscatter_mul * segment)
        transmittance_before = torch.cat(
            [torch.ones_like(scatter_segment_t[:1]), torch.cumprod(scatter_segment_t[:-1], dim=0)],
            dim=0,
        )
        heterogeneous_scatter = (
            heterogeneous_ambient
            * (1.0 - scatter_segment_t)
            * transmittance_before
        ).sum(dim=0)
        homogeneous_scatter_segment_t = torch.exp(
            -backscatter.unsqueeze(0) * segment
        ).expand_as(backscatter_mul)
        homogeneous_transmittance_before = torch.cat(
            [
                torch.ones_like(homogeneous_scatter_segment_t[:1]),
                torch.cumprod(homogeneous_scatter_segment_t[:-1], dim=0),
            ],
            dim=0,
        )
        homogeneous_scatter = (
            ambient.unsqueeze(0)
            * (1.0 - homogeneous_scatter_segment_t)
            * homogeneous_transmittance_before
        ).sum(dim=0)
        base_scatter = ambient * (
            1.0 - torch.exp(-backscatter * normalized_distance)
        )
        scatter = base_scatter + (heterogeneous_scatter - homogeneous_scatter)
        if observe_medium:
            self.field.observe_latents(raw.reshape(-1, 4), raw.new_ones(raw.numel() // 4))
        effective_backscatter = backscatter * backscatter_mul.mean(dim=0)
        effective_attenuation = attenuation * attenuation_mul.mean(dim=0)
        effective_ambient = heterogeneous_ambient.mean(dim=0)
        return {
            "direct_transmittance": direct,
            "backscatter_radiance": scatter,
            "ambient": effective_ambient,
            "backscatter": effective_backscatter,
            "attenuation": effective_attenuation,
            "optical_depth": effective_backscatter * normalized_distance,
            "field_latent": raw.mean(dim=0),
        }

    def transform_projected_colors(
        self,
        intrinsic_colors: torch.Tensor,
        means: torch.Tensor,
        camera_to_world: torch.Tensor,
        radii: torch.Tensor,
        camera_ids: Optional[torch.Tensor] = None,
        gaussian_ids: Optional[torch.Tensor] = None,
        include_intrinsic: bool = True,
    ) -> torch.Tensor:
        if camera_ids is not None or gaussian_ids is not None:
            raise NotImplementedError("Packed transport is not used by SeaFree-GS")
        transformed = []
        detached_means = means.detach()
        for camera_index in range(intrinsic_colors.shape[0]):
            intrinsic = intrinsic_colors[camera_index]
            visible_ids = torch.where(radii[camera_index] > 0)[0]
            degraded = intrinsic.clone()
            if visible_ids.numel() > 0:
                origin = camera_to_world[camera_index, :3, 3].detach().to(detached_means)
                vectors = detached_means[visible_ids] - origin
                distances = torch.linalg.vector_norm(vectors, dim=-1).clamp_min(1e-5)
                directions = vectors / distances[:, None]
                count = visible_ids.shape[0]
                ambient = self.field.ambient_anchor.to(directions).expand(count, -1)
                backscatter = directions.new_tensor([0.08, 0.16, 0.22]).expand(count, -1)
                attenuation = directions.new_tensor([0.30, 0.16, 0.08]).expand(count, -1)
                path = self.integrate_base_paths(
                    ambient, backscatter, attenuation, origin, directions, distances
                )
                degraded_visible = (
                    intrinsic[visible_ids] * path["direct_transmittance"]
                    + path["backscatter_radiance"]
                )
                degraded = degraded.index_copy(0, visible_ids, degraded_visible)
            channels = [degraded, intrinsic] if include_intrinsic else [degraded]
            transformed.append(torch.cat(channels, dim=-1))
        return torch.stack(transformed, dim=0)

    def integrate_background_from_base(
        self,
        ambient: torch.Tensor,
        backscatter: torch.Tensor,
        attenuation: torch.Tensor,
        directions: torch.Tensor,
        camera_origin: torch.Tensor,
        enabled: bool,
        prefix: Optional[TransportPrefix] = None,
    ) -> Dict[str, torch.Tensor]:
        del directions, camera_origin
        height, width = ambient.shape[:2]
        base = {
            "backscatter_radiance": ambient,
            "ambient": ambient,
            "backscatter": backscatter,
            "attenuation": attenuation,
        }
        if not enabled or prefix is None:
            return base
        prefix_backscatter, prefix_attenuation = prefix["backscatter"], prefix["attenuation"]
        mean_backscatter = prefix["mean_backscatter_multiplier"]
        mean_attenuation = prefix["mean_attenuation_multiplier"]
        assert all(
            isinstance(value, torch.Tensor)
            for value in (
                prefix_backscatter,
                prefix_attenuation,
                mean_backscatter,
                mean_attenuation,
            )
        )
        scatter_residual = prefix["scatter_residual"]
        scatter_transmittance_residual = prefix["scatter_transmittance_residual"]
        prefix_ambient = prefix["ambient"]
        assert isinstance(scatter_residual, torch.Tensor)
        assert isinstance(scatter_transmittance_residual, torch.Tensor)
        assert isinstance(prefix_ambient, torch.Tensor)
        correction = (
            scatter_residual[-1]
            + scatter_transmittance_residual[-1] * prefix_ambient
        )
        low_backscatter = prefix_backscatter * mean_backscatter
        low_attenuation = prefix_attenuation * mean_attenuation
        return {
            "backscatter_radiance": (
                ambient + self._upsample(correction, height, width)
            ).clamp(0.0, 1.0),
            "ambient": ambient,
            "backscatter": self._upsample(low_backscatter, height, width),
            "attenuation": self._upsample(low_attenuation, height, width),
        }

    def render_background(
        self,
        height: int,
        width: int,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        def fixed_properties(directions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            count = directions.shape[0]
            return (
                self.field.ambient_anchor.to(directions).expand(count, -1),
                directions.new_tensor([0.08, 0.16, 0.22]).expand(count, -1),
                directions.new_tensor([0.30, 0.16, 0.08]).expand(count, -1),
                directions.new_zeros((count, 9)),
            )

        prefix = self.build_frustum_prefix(
            height,
            width,
            intrinsics,
            camera_to_world,
            dtype,
            fixed_properties,
            enabled=True,
        )
        return {
            "backscatter_radiance": self._upsample(
                prefix["scatter"][-1], height, width
            ).unsqueeze(0),
            "ambient": self._upsample(prefix["ambient"], height, width).unsqueeze(0),
            "backscatter": self._upsample(
                prefix["backscatter"] * prefix["mean_backscatter_multiplier"], height, width
            ).unsqueeze(0),
            "attenuation": self._upsample(
                prefix["attenuation"] * prefix["mean_attenuation_multiplier"], height, width
            ).unsqueeze(0),
        }

    def diagnostics(
        self,
        expected_depth: torch.Tensor,
        alpha: torch.Tensor,
        background: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        surface_depth = self.field.normalize_distance(
            torch.nan_to_num(expected_depth, nan=0.0).clamp_min(0.0), self.depth_scale
        ).clamp(0.0, self.far_distance)
        ray_depth = torch.where(
            alpha > 1e-5, surface_depth, torch.full_like(surface_depth, self.far_distance)
        )
        return {
            "direct_transmittance": torch.exp(-background["attenuation"] * surface_depth),
            "optical_depth": background["backscatter"] * ray_depth,
        }
