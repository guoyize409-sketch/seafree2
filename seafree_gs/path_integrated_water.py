"""Anchor-constrained segment-integrated underwater transport."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class CompactHeterogeneousWaterField(nn.Module):
    """Three low-frequency physical latents around the validated global WPP.

    The channels are shared RGB multipliers for ambient intensity, turbidity,
    and the backscatter/attenuation ratio.  Shared multipliers preserve the
    spectral ordering learned by the directional WPP and cannot invent an
    independent water colour at every location.
    """

    def __init__(self, resolution: int = 24) -> None:
        super().__init__()
        self.resolution = max(int(resolution), 4)
        self.planes = nn.Parameter(torch.zeros(3, 3, self.resolution, self.resolution))
        self.register_buffer("latent_running_mean", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("latent_observation_mass", torch.zeros((), dtype=torch.float32))
        self.register_buffer("ambient_anchor", torch.tensor([0.18, 0.38, 0.48], dtype=torch.float32))
        self.register_buffer("ambient_observation_mass", torch.zeros((), dtype=torch.float32))
        self.register_buffer("aabb_min", torch.full((3,), -1.0, dtype=torch.float32))
        self.register_buffer("aabb_max", torch.full((3,), 1.0, dtype=torch.float32))
        self.register_buffer("scene_scale", torch.tensor(10.0, dtype=torch.float32))

    @torch.no_grad()
    def set_bounds(self, points: torch.Tensor) -> None:
        if points.numel() == 0:
            return
        values = points.detach().float()
        if values.shape[0] >= 64:
            lower = torch.quantile(values, 0.02, dim=0)
            upper = torch.quantile(values, 0.98, dim=0)
        else:
            lower = values.amin(dim=0)
            upper = values.amax(dim=0)
        extent = (upper - lower).clamp_min(1e-3)
        padding = 0.05 * extent
        self.aabb_min.copy_((lower - padding).to(self.aabb_min))
        self.aabb_max.copy_((upper + padding).to(self.aabb_max))
        self.scene_scale.copy_(
            (0.35 * torch.linalg.vector_norm(extent)).clamp_min(1e-3).to(self.scene_scale)
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
        return ((positions - lower) / extent * 2.0 - 1.0).clamp(-1.0, 1.0)

    def _sample_planes(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.numel() == 0:
            return positions.new_zeros((*positions.shape[:-1], 3))
        original_shape = positions.shape[:-1]
        normalized = self.normalize_positions(positions.reshape(-1, 3)).float()
        coordinates = torch.stack(
            [
                normalized[:, [0, 1]],
                normalized[:, [0, 2]],
                normalized[:, [1, 2]],
            ],
            dim=0,
        ).unsqueeze(1)
        sampled = F.grid_sample(
            self.planes.float(),
            coordinates,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        raw = sampled.squeeze(2).mean(dim=0).transpose(0, 1)
        return raw.reshape(*original_shape, 3).to(positions)

    def residual_raw(
        self, positions: torch.Tensor, directions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del directions
        return self._sample_planes(positions)

    @torch.no_grad()
    def observe_latents(
        self, raw_latents: torch.Tensor, weights: torch.Tensor, ema: float = 0.98
    ) -> None:
        if raw_latents.numel() == 0:
            return
        raw = raw_latents.detach().float().reshape(-1, 3)
        weight = weights.detach().float().reshape(-1, 1).clamp(0.0, 1.0)
        if raw.shape[0] != weight.shape[0]:
            return
        denominator = weight.sum()
        observed = (raw * weight).sum(dim=0) / denominator.clamp_min(1e-5)
        valid = (denominator > 1e-3).to(self.latent_running_mean)
        keep = min(max(float(ema), 0.0), 0.9999)
        update = valid * (1.0 - keep)
        self.latent_running_mean.lerp_(observed.to(self.latent_running_mean), update)
        next_mass = self.latent_observation_mass * keep + denominator.to(
            self.latent_observation_mass
        ) * (1.0 - keep)
        self.latent_observation_mass.copy_(
            torch.where(valid > 0, next_mass, self.latent_observation_mass)
        )

    def multipliers(
        self,
        positions: torch.Tensor,
        spatial_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self._sample_planes(positions)
        centered = raw - self.latent_running_mean.detach().to(raw)
        if spatial_gate is not None:
            gate = spatial_gate.to(centered)
            while gate.dim() < centered.dim():
                gate = gate.unsqueeze(-1)
            centered = centered * gate
        bounded = torch.tanh(centered)
        ambient_multiplier = torch.exp(0.10 * bounded[..., 0:1])
        density_multiplier = torch.exp(0.24 * bounded[..., 1:2])
        ratio_multiplier = torch.exp(0.18 * bounded[..., 2:3])
        return ambient_multiplier, density_multiplier, ratio_multiplier, raw

    def apply_to_base(
        self,
        ambient: torch.Tensor,
        backscatter: torch.Tensor,
        attenuation: torch.Tensor,
        positions: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
        ambient_scale: float = 0.04,
        coefficient_log_scale: float = 0.20,
        spatial_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del directions, ambient_scale, coefficient_log_scale
        ambient_mul, density_mul, ratio_mul, raw = self.multipliers(
            positions, spatial_gate=spatial_gate
        )
        return (
            (ambient * ambient_mul).clamp(0.0, 1.0),
            backscatter * density_mul * ratio_mul,
            attenuation * density_mul,
            raw,
        )

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
        decisive = (weight > 0.85).to(weight) * weight
        denominator = decisive.sum()
        observed = (rgb * decisive).sum(dim=(0, 1)) / denominator.clamp_min(1e-5)
        valid = denominator > 16.0
        keep = min(max(float(ema), 0.0), 0.9999)
        update = valid.to(self.ambient_anchor) * (1.0 - keep)
        self.ambient_anchor.lerp_(observed.to(self.ambient_anchor), update)
        updated_mass = self.ambient_observation_mass * keep + denominator.to(
            self.ambient_observation_mass
        ) * (1.0 - keep)
        self.ambient_observation_mass.copy_(
            torch.where(valid, updated_mass, self.ambient_observation_mass)
        )


class PathIntegratedWaterRenderer(nn.Module):
    """Two-segment transport that is analytically anchored to the Base formula."""

    def __init__(
        self,
        field: CompactHeterogeneousWaterField,
        query_downscale: int = 8,
        depth_scale: float = 10.0,
        far_distance: float = 4.0,
    ) -> None:
        super().__init__()
        self.field = field
        self.query_downscale = max(int(query_downscale), 1)
        self.depth_scale = max(float(depth_scale), 1e-4)
        self.far_distance = max(float(far_distance), 1e-3)

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
                "field_latent": ambient.new_zeros((*ambient.shape[:-1], 3)),
            }
        if origins.dim() == 1:
            origins = origins.unsqueeze(0).expand_as(directions)
        distance_column = distances[:, None]
        sample_positions = torch.stack(
            [
                origins + 0.25 * distance_column * directions,
                origins + 0.75 * distance_column * directions,
            ],
            dim=0,
        )
        if spatial_gate is None:
            segment_gate = None
        else:
            segment_gate = spatial_gate.reshape(1, -1).expand(2, -1)
        ambient_mul, density_mul, ratio_mul, raw = self.field.multipliers(
            sample_positions, spatial_gate=segment_gate
        )
        if observe_medium and spatial_gate is not None:
            self.field.observe_latents(
                raw.reshape(-1, 3),
                spatial_gate.reshape(1, -1).expand(2, -1).reshape(-1),
            )

        normalized = self.field.normalize_distance(distances, self.depth_scale).clamp(
            0.0, self.far_distance
        )
        segment_distance = 0.5 * normalized[:, None]
        density_mean = 0.5 * (density_mul[0] + density_mul[1])
        direct = torch.exp(-attenuation * normalized[:, None] * density_mean)

        tau = backscatter.unsqueeze(0) * segment_distance.unsqueeze(0)
        tau = tau * density_mul * ratio_mul
        segment_scatter = 1.0 - torch.exp(-tau)
        ambient_segments = ambient.unsqueeze(0) * ambient_mul
        heterogeneous = ambient_segments[0] * segment_scatter[0]
        heterogeneous = heterogeneous + torch.exp(-tau[0]) * ambient_segments[1] * segment_scatter[1]

        homogeneous_tau = backscatter * segment_distance
        homogeneous_segment = 1.0 - torch.exp(-homogeneous_tau)
        homogeneous = ambient * homogeneous_segment
        homogeneous = homogeneous + torch.exp(-homogeneous_tau) * ambient * homogeneous_segment
        base_backscatter = ambient * (1.0 - torch.exp(-backscatter * normalized[:, None]))
        radiance = base_backscatter + (heterogeneous - homogeneous)

        effective_backscatter = backscatter * (
            0.5 * (density_mul[0] * ratio_mul[0] + density_mul[1] * ratio_mul[1])
        )
        effective_attenuation = attenuation * density_mean
        effective_ambient = ambient * 0.5 * (ambient_mul[0] + ambient_mul[1])
        return {
            "direct_transmittance": direct,
            "backscatter_radiance": radiance,
            "ambient": effective_ambient,
            "backscatter": effective_backscatter,
            "attenuation": effective_attenuation,
            "optical_depth": effective_backscatter * normalized[:, None],
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
        camera_dirs = torch.stack([(xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx)], dim=-1)
        camera_dirs = F.normalize(camera_dirs, dim=-1)
        world_dirs = camera_dirs @ camera_to_world[0, :3, :3].transpose(0, 1)
        return F.normalize(world_dirs, dim=-1), low_h, low_w

    @staticmethod
    def _upsample(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
        low = value.permute(2, 0, 1).unsqueeze(0)
        return F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)

    def integrate_background_from_base(
        self,
        ambient: torch.Tensor,
        backscatter: torch.Tensor,
        attenuation: torch.Tensor,
        directions: torch.Tensor,
        camera_origin: torch.Tensor,
        enabled: bool,
    ) -> Dict[str, torch.Tensor]:
        height, width = ambient.shape[:2]
        base = {
            "backscatter_radiance": ambient,
            "ambient": ambient,
            "backscatter": backscatter,
            "attenuation": attenuation,
        }
        if not enabled:
            return base
        low_h = max(1, (height + self.query_downscale - 1) // self.query_downscale)
        low_w = max(1, (width + self.query_downscale - 1) // self.query_downscale)

        def downsample(value: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                value.permute(2, 0, 1).unsqueeze(0),
                size=(low_h, low_w),
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0)

        low_a = downsample(ambient)
        low_b = downsample(backscatter)
        low_s = downsample(attenuation)
        low_d = F.normalize(downsample(directions), dim=-1)
        flat_d = low_d.reshape(-1, 3)
        count = flat_d.shape[0]
        distances = flat_d.new_full((count,), self.depth_scale * self.far_distance)
        path = self.integrate_base_paths(
            low_a.reshape(-1, 3),
            low_b.reshape(-1, 3),
            low_s.reshape(-1, 3),
            camera_origin.to(flat_d),
            flat_d,
            distances,
        )
        normalized = self.field.normalize_distance(distances, self.depth_scale)[:, None]
        homogeneous = low_a.reshape(-1, 3) * (
            1.0 - torch.exp(-low_b.reshape(-1, 3) * normalized)
        )
        correction = (path["backscatter_radiance"] - homogeneous).reshape(low_h, low_w, 3)
        result = {
            "backscatter_radiance": (
                ambient + self._upsample(correction, height, width)
            ).clamp(0.0, 1.0),
            "ambient": self._upsample(path["ambient"].reshape(low_h, low_w, 3), height, width),
            "backscatter": self._upsample(path["backscatter"].reshape(low_h, low_w, 3), height, width),
            "attenuation": self._upsample(path["attenuation"].reshape(low_h, low_w, 3), height, width),
        }
        return result

    def render_background(
        self,
        height: int,
        width: int,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        directions, low_h, low_w = self._low_resolution_rays(
            height, width, intrinsics, camera_to_world, dtype
        )
        flat = directions.reshape(-1, 3)
        count = flat.shape[0]
        ambient = self.field.ambient_anchor.to(flat).expand(count, -1)
        backscatter = flat.new_tensor([0.08, 0.16, 0.22]).expand(count, -1)
        attenuation = flat.new_tensor([0.30, 0.16, 0.08]).expand(count, -1)
        distance = flat.new_full((count,), self.depth_scale * self.far_distance)
        path = self.integrate_base_paths(
            ambient,
            backscatter,
            attenuation,
            camera_to_world[0, :3, 3].detach().to(flat),
            flat,
            distance,
        )
        return {
            key: self._upsample(value.reshape(low_h, low_w, 3), height, width).unsqueeze(0)
            for key, value in path.items()
            if value.shape[-1] == 3
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
        ray_depth = torch.where(alpha > 1e-5, surface_depth, torch.full_like(surface_depth, self.far_distance))
        return {
            "direct_transmittance": torch.exp(-background["attenuation"] * surface_depth),
            "optical_depth": background["backscatter"] * ray_depth,
        }
