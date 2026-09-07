"""Physics-identifiable surface/medium ownership for underwater Gaussians."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


class PhysicsIdentifiableDualCarrier(nn.Module):
    """Online dual-stream ownership inferred from occlusion-aware evidence.

    A Gaussian keeps its full opacity in the underwater stream.  The inferred
    surface fraction is used only for intrinsic radiance and clear rendering;
    the complementary medium fraction is rendered with water radiance.  This
    avoids destructive geometry edits while preventing water carriers from
    exposing learned SH colour in the clear view.
    """

    UNCERTAIN = 0
    SURFACE = 1
    MEDIUM = 2
    STATE_KEYS = (
        "carrier_medium_evidence",
        "carrier_surface_evidence",
        "carrier_observation_mass",
        "carrier_window_medium_evidence",
        "carrier_window_surface_evidence",
        "carrier_window_observation_mass",
        "carrier_view_mask",
        "carrier_lifetime_view_mask",
        "carrier_angular_coverage",
        "carrier_lifetime_coverage",
        "carrier_direction_sum",
        "carrier_parallax_evidence",
        "carrier_candidate_state",
        "carrier_candidate_streak",
        "carrier_certified_state",
        "carrier_sfm_reliability",
        "carrier_window_view_ids",
        "carrier_topology_medium_support",
        "carrier_topology_surface_support",
        "carrier_topology_confidence",
    )

    def __init__(
        self,
        num_points: int,
        device: torch.device,
        update_every: int = 100,
        min_distinct_views: int = 2,
        num_train_views: int = 1,
        warmup_steps: int = 4000,
        epoch_steps: int = 0,
        min_certified_epochs: int = 2,
        depth_scale: float = 10.0,
        chunk_size: int = 262144,
        evidence_decay: float = 0.97,
        min_observation_mass: float = 1.25,
        sfm_reliability: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.update_every = max(int(update_every), 1)
        self.min_distinct_views = max(2, int(min_distinct_views))
        self.tracked_view_slots = min(max(self.min_distinct_views, 4), 8)
        self.num_train_views = max(int(num_train_views), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        inferred_window = self.num_train_views * self.update_every
        self.coverage_window_steps = max(
            int(epoch_steps) if epoch_steps > 0 else min(max(inferred_window, 2000), 6000),
            self.update_every,
        )
        self.min_candidate_streak = max(2, int(min_certified_epochs))
        self.depth_scale = max(float(depth_scale), 1e-4)
        self.chunk_size = max(int(chunk_size), 16384)
        self.evidence_decay = min(max(float(evidence_decay), 0.90), 0.999)
        self.min_observation_mass = max(float(min_observation_mass), 1e-4)
        self._window_index_value = -1
        self._register_state(num_points, device, sfm_reliability)

    def _register_state(
        self,
        num_points: int,
        device: torch.device,
        sfm_reliability: Optional[torch.Tensor],
    ) -> None:
        if sfm_reliability is None or sfm_reliability.numel() != num_points:
            sfm_u8 = torch.zeros(num_points, device=device, dtype=torch.uint8)
        else:
            sfm_u8 = (
                sfm_reliability.detach()
                .to(device=device, dtype=torch.float32)
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
            )
        sfm = sfm_u8.float() / 255.0
        self.register_buffer("medium_evidence", torch.full((num_points,), 0.5, device=device))
        self.register_buffer("surface_evidence", 0.5 + 1.5 * sfm)
        self.register_buffer("observation_mass", torch.zeros(num_points, device=device))
        self.register_buffer(
            "window_medium_evidence", torch.zeros(num_points, device=device, dtype=torch.float16)
        )
        self.register_buffer(
            "window_surface_evidence", torch.zeros(num_points, device=device, dtype=torch.float16)
        )
        self.register_buffer(
            "window_observation_mass", torch.zeros(num_points, device=device, dtype=torch.float16)
        )
        self.register_buffer("view_mask", torch.zeros(num_points, device=device, dtype=torch.int64))
        self.register_buffer(
            "lifetime_view_mask", torch.zeros(num_points, device=device, dtype=torch.int64)
        )
        self.register_buffer(
            "angular_coverage", torch.zeros(num_points, device=device, dtype=torch.uint8)
        )
        self.register_buffer(
            "lifetime_coverage", torch.zeros(num_points, device=device, dtype=torch.uint8)
        )
        self.register_buffer(
            "direction_sum", torch.zeros(num_points, 3, device=device, dtype=torch.float16)
        )
        self.register_buffer(
            "parallax_evidence", torch.zeros(num_points, device=device, dtype=torch.float16)
        )
        self.register_buffer(
            "candidate_state", torch.zeros(num_points, device=device, dtype=torch.uint8)
        )
        self.register_buffer(
            "candidate_streak", torch.zeros(num_points, device=device, dtype=torch.uint8)
        )
        self.register_buffer(
            "certified_state", torch.zeros(num_points, device=device, dtype=torch.uint8)
        )
        self.register_buffer("sfm_reliability", sfm_u8)
        self.register_buffer(
            "window_view_ids",
            torch.full(
                (num_points, self.tracked_view_slots),
                -1,
                device=device,
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "topology_medium_support",
            torch.full((num_points,), 0.5, device=device, dtype=torch.float16),
        )
        self.register_buffer(
            "topology_surface_support",
            (0.5 + 0.5 * sfm).to(torch.float16),
        )
        self.register_buffer(
            "topology_confidence", torch.zeros(num_points, device=device, dtype=torch.float16)
        )
        self.register_buffer(
            "adaptive_medium_anchor", torch.tensor(0.56, device=device, dtype=torch.float32)
        )
        self.register_buffer("window_index", torch.full((), -1, device=device, dtype=torch.int32))

    @property
    def num_points(self) -> int:
        return int(self.surface_evidence.shape[0])

    # Compatibility aliases used by existing metrics and tests.
    @property
    def freewater_evidence(self) -> torch.Tensor:
        return self.medium_evidence

    @property
    def scene_evidence(self) -> torch.Tensor:
        return self.surface_evidence

    @property
    def haze_evidence(self) -> torch.Tensor:
        return self.probabilities()["uncertainty"]

    @property
    def empty_evidence(self) -> torch.Tensor:
        return self.medium_evidence

    @property
    def water_evidence(self) -> torch.Tensor:
        return self.medium_evidence

    @property
    def distinct_views(self) -> torch.Tensor:
        return self.angular_coverage

    @property
    def epoch_index(self) -> torch.Tensor:
        return self.window_index

    def resize(self, num_points: int, device: torch.device) -> None:
        if self.num_points == num_points and self.surface_evidence.device == device:
            return
        sfm = None
        if self.sfm_reliability.numel() == num_points:
            sfm = self.sfm_reliability.float() / 255.0
        replacement = PhysicsIdentifiableDualCarrier(
            num_points=num_points,
            device=device,
            update_every=self.update_every,
            min_distinct_views=self.min_distinct_views,
            num_train_views=self.num_train_views,
            warmup_steps=self.warmup_steps,
            epoch_steps=self.coverage_window_steps,
            min_certified_epochs=self.min_candidate_streak,
            depth_scale=self.depth_scale,
            chunk_size=self.chunk_size,
            evidence_decay=self.evidence_decay,
            min_observation_mass=self.min_observation_mass,
            sfm_reliability=sfm,
        )
        for name, value in replacement.named_buffers():
            setattr(self, name, value)
        self._window_index_value = -1

    def _state_values(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.medium_evidence,
            self.surface_evidence,
            self.observation_mass,
            self.window_medium_evidence,
            self.window_surface_evidence,
            self.window_observation_mass,
            self.view_mask,
            self.lifetime_view_mask,
            self.angular_coverage,
            self.lifetime_coverage,
            self.direction_sum,
            self.parallax_evidence,
            self.candidate_state,
            self.candidate_streak,
            self.certified_state,
            self.sfm_reliability,
            self.window_view_ids,
            self.topology_medium_support,
            self.topology_surface_support,
            self.topology_confidence,
        )

    def bind_strategy_state(self, state: Dict[str, torch.Tensor]) -> None:
        for key, value in zip(self.STATE_KEYS, self._state_values()):
            state[key] = value

    def sync_from_strategy_state(self, state: Dict[str, torch.Tensor]) -> None:
        values = [state.get(key) for key in self.STATE_KEYS]
        if any(value is None for value in values):
            self.bind_strategy_state(state)
            return
        typed = tuple(values)  # type: ignore[arg-type]
        current = self._state_values()
        if any(
            value.device != old.device or value.dtype != old.dtype
            for value, old in zip(typed, current)
        ):
            self.bind_strategy_state(state)
            return
        (
            self.medium_evidence,
            self.surface_evidence,
            self.observation_mass,
            self.window_medium_evidence,
            self.window_surface_evidence,
            self.window_observation_mass,
            self.view_mask,
            self.lifetime_view_mask,
            self.angular_coverage,
            self.lifetime_coverage,
            self.direction_sum,
            self.parallax_evidence,
            self.candidate_state,
            self.candidate_streak,
            self.certified_state,
            self.sfm_reliability,
            self.window_view_ids,
            self.topology_medium_support,
            self.topology_surface_support,
            self.topology_confidence,
        ) = typed

    @torch.no_grad()
    def _reset_window(self) -> None:
        self.window_medium_evidence.zero_()
        self.window_surface_evidence.zero_()
        self.window_observation_mass.zero_()
        self.view_mask.zero_()
        self.angular_coverage.zero_()
        self.direction_sum.zero_()
        self.parallax_evidence.zero_()
        self.window_view_ids.fill_(-1)

    def _posterior_components(self, include_pending: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        medium = self.medium_evidence.float()
        surface = self.surface_evidence.float()
        mass = self.observation_mass.float()
        if include_pending:
            medium = medium + self.window_medium_evidence.float()
            surface = surface + self.window_surface_evidence.float()
            mass = mass + self.window_observation_mass.float()
        return medium.clamp_min(1e-5), surface.clamp_min(1e-5), mass

    @torch.no_grad()
    def _update_topology_support(
        self,
        means: Optional[torch.Tensor],
        opacity: Optional[torch.Tensor],
        resolution: int = 24,
    ) -> None:
        """Propagate ownership only between Gaussians sharing a local 3D neighbourhood."""
        if means is None or means.numel() == 0 or means.shape[0] < self.num_points:
            return
        points = means[: self.num_points].detach().float()
        finite = torch.isfinite(points).all(dim=-1)
        if not bool(finite.any()):
            return
        lower = points[finite].amin(dim=0)
        upper = points[finite].amax(dim=0)
        extent = (upper - lower).clamp_min(1e-4)
        unit = (points - lower) / extent
        inside = finite & ((unit >= 0.0) & (unit <= 1.0)).all(dim=-1)
        voxel = (unit.clamp(0.0, 1.0) * (resolution - 1)).round().to(torch.int64)
        flat_index = (
            voxel[:, 2] * resolution * resolution + voxel[:, 1] * resolution + voxel[:, 0]
        )
        if opacity is None or opacity.numel() < self.num_points:
            optical_weight = torch.ones(self.num_points, device=points.device)
        else:
            alpha = opacity[: self.num_points].detach().float().reshape(-1).clamp(0.0, 1.0 - 1e-6)
            optical_weight = -torch.log1p(-alpha)
        medium, surface, mass = self._posterior_components(include_pending=False)
        posterior_medium = medium / (medium + surface)
        evidence_confidence = (1.0 - torch.exp(-mass / self.min_observation_mass)).clamp(0.0, 1.0)
        weight = optical_weight * evidence_confidence * inside.to(optical_weight)

        voxel_count = resolution**3
        medium_grid = torch.zeros(voxel_count, device=points.device)
        surface_grid = torch.zeros_like(medium_grid)
        total_grid = torch.zeros_like(medium_grid)
        medium_grid.scatter_add_(0, flat_index, weight * posterior_medium)
        surface_grid.scatter_add_(0, flat_index, weight * (1.0 - posterior_medium))
        total_grid.scatter_add_(0, flat_index, weight)
        shape = (1, 1, resolution, resolution, resolution)
        medium_grid = torch.nn.functional.avg_pool3d(medium_grid.reshape(shape), 3, 1, 1)
        surface_grid = torch.nn.functional.avg_pool3d(surface_grid.reshape(shape), 3, 1, 1)
        total_grid = torch.nn.functional.avg_pool3d(total_grid.reshape(shape), 3, 1, 1)
        medium_local = medium_grid.reshape(-1)[flat_index]
        surface_local = surface_grid.reshape(-1)[flat_index]
        total_local = total_grid.reshape(-1)[flat_index]
        occupied = total_grid[total_grid > 0.0]
        reference_mass = occupied.mean() if occupied.numel() > 0 else total_local.new_tensor(1.0)
        confidence = (total_local / (total_local + reference_mass.clamp_min(1e-5))).clamp(0.0, 1.0)
        ratio = medium_local / (medium_local + surface_local).clamp_min(1e-5)
        ratio = torch.where(total_local > 0.0, ratio, torch.full_like(ratio, 0.5))
        self.topology_medium_support.lerp_(ratio.to(self.topology_medium_support), 0.35)
        self.topology_surface_support.lerp_((1.0 - ratio).to(self.topology_surface_support), 0.35)
        self.topology_confidence.lerp_(confidence.to(self.topology_confidence), 0.35)

    @torch.no_grad()
    def _finalize_window(
        self,
        means: Optional[torch.Tensor] = None,
        opacity: Optional[torch.Tensor] = None,
    ) -> None:
        """Commit one multi-view window to the persistent Beta posterior."""
        for start in range(0, self.num_points, self.chunk_size):
            end = min(start + self.chunk_size, self.num_points)
            window_mass = self.window_observation_mass[start:end].float()
            window_total = (
                self.window_medium_evidence[start:end].float()
                + self.window_surface_evidence[start:end].float()
            ).clamp_min(1e-5)
            window_medium = self.window_medium_evidence[start:end].float() / window_total
            coverage = self.angular_coverage[start:end]
            has_evidence = (coverage > 0) & (window_mass > 1e-3)
            minimum_window_views = min(3, self.min_distinct_views)
            decisive_window = (
                (coverage >= minimum_window_views)
                & (self.lifetime_coverage[start:end] >= self.min_distinct_views)
                & (window_mass >= 0.5 * self.min_observation_mass)
            )
            parallax_count = (coverage.to(torch.float32) - 1.0).clamp_min(1.0)
            mean_parallax = self.parallax_evidence[start:end].float() / parallax_count
            parallax_quality = ((mean_parallax - 5e-4) / 0.02).clamp(0.0, 1.0)
            effective_mass = window_mass.clamp(max=8.0) * (0.60 + 0.40 * parallax_quality)

            keep = torch.where(
                has_evidence,
                effective_mass.new_full((), self.evidence_decay),
                torch.ones_like(effective_mass),
            )
            self.medium_evidence[start:end].mul_(keep).add_(
                has_evidence * effective_mass * window_medium
            )
            self.surface_evidence[start:end].mul_(keep).add_(
                has_evidence * effective_mass * (1.0 - window_medium)
            )
            self.observation_mass[start:end].add_(has_evidence * effective_mass).clamp_(max=255.0)

            posterior_medium = self.medium_evidence[start:end] / (
                self.medium_evidence[start:end] + self.surface_evidence[start:end]
            ).clamp_min(1e-5)
            sfm = self.sfm_reliability[start:end].float() / 255.0
            medium_threshold = self.adaptive_medium_anchor + 0.035
            surface_threshold = torch.minimum(
                self.adaptive_medium_anchor - 0.07,
                self.adaptive_medium_anchor.new_tensor(0.48),
            )
            medium_candidate = (
                decisive_window
                & (window_medium >= self.adaptive_medium_anchor - 0.02)
                & (posterior_medium >= medium_threshold)
                & ((sfm < 0.80) | (posterior_medium >= medium_threshold + 0.12))
            )
            surface_candidate = decisive_window & (
                ((window_medium <= surface_threshold + 0.04) & (posterior_medium <= surface_threshold))
                | (window_medium <= 0.30)
                | ((sfm >= 0.82) & (posterior_medium <= self.adaptive_medium_anchor + 0.02))
            )
            candidate = torch.zeros_like(self.candidate_state[start:end])
            candidate[surface_candidate] = self.SURFACE
            candidate[medium_candidate & ~surface_candidate] = self.MEDIUM

            previous_candidate = self.candidate_state[start:end]
            previous_streak = self.candidate_streak[start:end]
            same = (candidate == previous_candidate) & (candidate != self.UNCERTAIN)
            next_streak = torch.where(
                same,
                (previous_streak.to(torch.int16) + 1).clamp_max(255).to(torch.uint8),
                (candidate != self.UNCERTAIN).to(torch.uint8),
            )
            next_candidate = torch.where(decisive_window, candidate, previous_candidate)
            next_streak = torch.where(decisive_window, next_streak, previous_streak)

            certified = self.certified_state[start:end]
            contradicted = (
                decisive_window
                & (certified != self.UNCERTAIN)
                & (candidate != self.UNCERTAIN)
                & (candidate != certified)
            )
            next_state = torch.where(contradicted, torch.zeros_like(certified), certified)
            committed = decisive_window & (next_streak >= self.min_candidate_streak)
            next_state = torch.where(committed, next_candidate, next_state)
            self.candidate_state[start:end].copy_(next_candidate)
            self.candidate_streak[start:end].copy_(next_streak)
            self.certified_state[start:end].copy_(next_state)

        medium, surface, mass = self._posterior_components(include_pending=False)
        posterior = medium / (medium + surface)
        adaptive_population = posterior[
            (mass >= 0.5 * self.min_observation_mass)
            & (self.sfm_reliability.float() < 0.90 * 255.0)
        ]
        if adaptive_population.numel() >= 8:
            quartiles = torch.quantile(
                adaptive_population,
                adaptive_population.new_tensor([0.25, 0.50, 0.75]),
            )
            robust_spread = (quartiles[2] - quartiles[0]).clamp_min(0.02)
            target_anchor = (quartiles[1] + 0.30 * robust_spread).clamp(0.50, 0.64)
            self.adaptive_medium_anchor.lerp_(target_anchor, 0.35)
        self._update_topology_support(means, opacity)

    @torch.no_grad()
    def finalize_pending_window(
        self,
        means: Optional[torch.Tensor] = None,
        opacity: Optional[torch.Tensor] = None,
    ) -> bool:
        """Commit the final partial training window exactly once before save/evaluation."""
        if not bool((self.window_observation_mass > 0).any()):
            return False
        self._finalize_window(means=means, opacity=opacity)
        self._reset_window()
        return True

    @torch.no_grad()
    def _roll_coverage_window(
        self,
        step: int,
        means: Optional[torch.Tensor] = None,
        opacity: Optional[torch.Tensor] = None,
    ) -> None:
        if step < self.warmup_steps:
            return
        window = (step - self.warmup_steps) // self.coverage_window_steps
        if window != self._window_index_value:
            if self._window_index_value >= 0:
                self._finalize_window(means=means, opacity=opacity)
            self._reset_window()
            self.window_index.fill_(window)
            self._window_index_value = window

    def probabilities(self) -> Dict[str, torch.Tensor]:
        medium, surface, observation_mass = self._posterior_components(include_pending=True)
        probability_medium = medium / (medium + surface)
        topology_total = (
            self.topology_medium_support.float() + self.topology_surface_support.float()
        ).clamp_min(1e-5)
        topology_medium = self.topology_medium_support.float() / topology_total
        topology_logit = torch.logit(topology_medium.clamp(1e-4, 1.0 - 1e-4))
        posterior_logit = torch.logit(probability_medium.clamp(1e-4, 1.0 - 1e-4))
        topology_mix = 0.25 * self.topology_confidence.float().clamp(0.0, 1.0)
        probability_medium = torch.sigmoid(posterior_logit + topology_mix * topology_logit)
        probability_surface = 1.0 - probability_medium
        geometry = torch.stack([probability_medium, probability_surface], dim=-1).clamp_min(1e-5)
        entropy = -(geometry * geometry.log()).sum(dim=-1) / geometry.new_tensor(2.0).log()
        observation_uncertainty = torch.exp(-observation_mass / self.min_observation_mass)
        uncertainty = torch.maximum(entropy, observation_uncertainty).clamp(0.0, 1.0)
        certified = self.certified_state != self.UNCERTAIN
        observed = (
            (self.lifetime_coverage >= self.min_distinct_views)
            & (observation_mass >= self.min_observation_mass)
        )
        zero = torch.zeros_like(probability_medium)
        return {
            "freewater": probability_medium,
            "medium": probability_medium,
            "haze": zero,
            "veil": zero,
            "scene": probability_surface,
            "surface": probability_surface,
            "uncertainty": uncertainty,
            "certified": certified,
            "observed": observed,
            "certified_medium": self.certified_state == self.MEDIUM,
            "certified_surface": self.certified_state == self.SURFACE,
        }

    def render_weights(self) -> Dict[str, torch.Tensor]:
        ownership = self.probabilities()
        medium_probability = ownership["medium"]
        _, _, observation_mass = self._posterior_components(include_pending=True)
        mass_confidence = 1.0 - torch.exp(-observation_mass / self.min_observation_mass)
        angular_confidence = (
            self.lifetime_coverage.float() / max(float(self.min_distinct_views), 1.0)
        ).clamp(0.0, 1.0)
        confidence = torch.sqrt((mass_confidence * (0.20 + 0.80 * angular_confidence)).clamp_min(0.0))
        lower = self.adaptive_medium_anchor - 0.04
        soft_medium = ((medium_probability - lower) / 0.16).clamp(0.0, 1.0)
        sfm_guard = 1.0 - 0.80 * (self.sfm_reliability.float() / 255.0)
        medium_fraction = (confidence * soft_medium * sfm_guard).clamp(0.0, 1.0)
        surface_fraction = 1.0 - medium_fraction
        surface_fraction = torch.where(
            ownership["certified_surface"], torch.ones_like(surface_fraction), surface_fraction
        )
        surface_fraction = torch.where(
            ownership["certified_medium"], torch.zeros_like(surface_fraction), surface_fraction
        )
        medium_fraction = 1.0 - surface_fraction
        return {
            **ownership,
            "clear_scene": surface_fraction,
            "underwater_geometry": torch.ones_like(surface_fraction),
            "surface_geometry": surface_fraction,
            "surface_fraction": surface_fraction,
            "medium_fraction": medium_fraction,
            "transport_confidence": medium_fraction,
            "freewater_hard": ownership["certified_medium"],
            "haze_hard": torch.zeros_like(ownership["certified_medium"]),
        }

    def probability(self) -> torch.Tensor:
        return self.render_weights()["clear_scene"]

    @staticmethod
    def split_optical_thickness(
        opacity: torch.Tensor,
        surface_fraction: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split extinction while exactly preserving total optical thickness."""
        alpha = opacity.clamp(0.0, 1.0 - 1e-6)
        q = surface_fraction.to(alpha).clamp(0.0, 1.0)
        tau = -torch.log1p(-alpha)
        surface_alpha = -torch.expm1(-q * tau)
        medium_alpha = -torch.expm1(-(1.0 - q) * tau)
        return surface_alpha, medium_alpha

    def lifecycle_masks(
        self,
        step: int,
        grow_start: int,
        prune_start: int,
        max_prune_fraction: float,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        del prune_start, max_prune_fraction
        hard_medium = self.render_weights()["freewater_hard"]
        return (hard_medium if step >= grow_start else None), None

    @staticmethod
    @torch.no_grad()
    def rasterize_responsibility_mass(
        info: Dict[str, torch.Tensor],
        medium: Optional[torch.Tensor] = None,
        surface: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        valid_image_mask: Optional[torch.Tensor] = None,
        freewater: Optional[torch.Tensor] = None,
        haze: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        required = ("means2d", "conics", "opacities", "isect_offsets", "flatten_ids")
        if any(info.get(key) is None for key in required) or not info["means2d"].is_cuda:
            return None
        medium = freewater if medium is None else medium
        surface = scene if surface is None else surface
        if medium is None or surface is None:
            return None
        if uncertainty is None:
            uncertainty = haze if haze is not None else 1.0 - torch.maximum(medium, surface)
        responsibilities = torch.cat([medium, surface, uncertainty], dim=-1)
        responsibilities = responsibilities.detach().float().clamp(0.0, 1.0)
        if valid_image_mask is not None:
            responsibilities *= valid_image_mask[..., :1].detach().to(responsibilities) > 0.5

        from gsplat.cuda._wrapper import rasterize_responsibilities

        return rasterize_responsibilities(
            means2d=info["means2d"],
            conics=info["conics"],
            opacities=info["opacities"],
            responsibilities=responsibilities.unsqueeze(0),
            image_width=int(info["width"]),
            image_height=int(info["height"]),
            tile_size=int(info["tile_size"]),
            isect_offsets=info["isect_offsets"],
            flatten_ids=info["flatten_ids"],
        )

    @torch.no_grad()
    def update_from_responsibilities(
        self,
        step: int,
        view_id: torch.Tensor,
        info: Dict[str, torch.Tensor],
        medium: Optional[torch.Tensor] = None,
        surface: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        contribution_mass: Optional[torch.Tensor] = None,
        responsibility_mass: Optional[torch.Tensor] = None,
        valid_image_mask: Optional[torch.Tensor] = None,
        freewater: Optional[torch.Tensor] = None,
        haze: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
        means: Optional[torch.Tensor] = None,
        camera_origin: Optional[torch.Tensor] = None,
        geometric_medium_support: Optional[torch.Tensor] = None,
        geometric_surface_support: Optional[torch.Tensor] = None,
    ) -> None:
        del contribution_mass, valid_image_mask, haze, medium, surface, uncertainty, freewater, scene
        if step < self.warmup_steps or step % self.update_every != 0:
            return
        projected_opacity = info.get("opacities")
        if projected_opacity is not None and projected_opacity.dim() > 1:
            projected_opacity = projected_opacity.reshape(-1)
        self._roll_coverage_window(step, means=means, opacity=projected_opacity)
        if responsibility_mass is None or responsibility_mass.numel() == 0:
            return
        means2d = info.get("means2d")
        radii = info.get("radii")
        if means2d is None or radii is None or means2d.dim() != 3 or means2d.shape[0] != 1:
            return
        exact = responsibility_mass.reshape(-1, 3).detach().float().clamp_min(0.0)
        n = min(self.num_points, means2d.shape[1], exact.shape[0])
        if n == 0:
            return

        current_view_id = view_id.detach().reshape(-1)[0].to(device=exact.device, dtype=torch.int64)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            weighted = exact[start:end]
            contribution = weighted.sum(dim=-1)
            visible_information = 1.0 - torch.exp(-contribution / 4.0)
            decisive = weighted[:, 0] + weighted[:, 1]
            decisive_fraction = decisive / contribution.clamp_min(1e-5)
            medium_share = weighted[:, 0] / decisive.clamp_min(1e-5)

            if (
                geometric_medium_support is not None
                and geometric_surface_support is not None
                and geometric_medium_support.numel() >= end
                and geometric_surface_support.numel() >= end
            ):
                geometry_medium = geometric_medium_support[start:end].detach().float().clamp(0.0, 1.0)
                geometry_surface = geometric_surface_support[start:end].detach().float().clamp(0.0, 1.0)
                geometry_total = (geometry_medium + geometry_surface).clamp(0.0, 1.0)
                geometry_share = geometry_medium / (
                    geometry_medium + geometry_surface
                ).clamp_min(1e-5)
                geometry_mix = 0.70 * geometry_total
                medium_share = torch.lerp(medium_share, geometry_share, geometry_mix)
                decisive_fraction = torch.maximum(decisive_fraction, 0.50 * geometry_total)
            surface_share = 1.0 - medium_share

            if means is not None and camera_origin is not None and means.shape[0] >= end:
                directions = means[start:end].detach() - camera_origin.detach().reshape(1, 3)
                directions = torch.nn.functional.normalize(directions, dim=-1, eps=1e-6)
            else:
                directions = None
            window_view_ids = self.window_view_ids[start:end]
            distinct_id = ~(window_view_ids == current_view_id.to(torch.int32)).any(dim=-1)
            coverage = self.angular_coverage[start:end].to(torch.int16)
            current_direction_sum = self.direction_sum[start:end].float()
            if directions is None:
                angular_novel = torch.ones_like(distinct_id)
                novelty = (coverage > 0).to(visible_information)
                direction_update = torch.zeros_like(current_direction_sum)
            else:
                mean_direction = torch.nn.functional.normalize(
                    current_direction_sum, dim=-1, eps=1e-6
                )
                cosine = (mean_direction * directions).sum(dim=-1).clamp(-1.0, 1.0)
                novelty = torch.where(
                    coverage > 0,
                    (1.0 - cosine).clamp(0.0, 2.0),
                    torch.ones_like(cosine),
                )
                angular_novel = (coverage == 0) | (novelty > 1e-5)
                direction_update = directions
            slot_available = coverage < self.tracked_view_slots
            new_window_view = (
                (visible_information > 1e-3)
                & distinct_id
                & angular_novel
                & slot_available
            )
            update_mass = visible_information * decisive_fraction * new_window_view.float()
            new_lifetime_view = new_window_view
            row = torch.arange(end - start, device=exact.device)
            insert_at = coverage.clamp(0, self.tracked_view_slots - 1).to(torch.int64)
            window_view_ids[row[new_window_view], insert_at[new_window_view]] = current_view_id.to(
                torch.int32
            )
            self.angular_coverage[start:end].copy_(
                (coverage + new_window_view.to(torch.int16)).clamp_max(255).to(torch.uint8)
            )
            lifetime_coverage = self.lifetime_coverage[start:end].to(torch.int16)
            self.lifetime_coverage[start:end].copy_(
                (lifetime_coverage + new_lifetime_view.to(torch.int16))
                .clamp_max(255)
                .to(torch.uint8)
            )

            self.direction_sum[start:end].copy_(
                (current_direction_sum + new_window_view[:, None] * direction_update).to(torch.float16)
            )
            self.parallax_evidence[start:end].add_(
                (new_window_view * novelty).to(torch.float16)
            )

            observed = update_mass > 0.0
            self.window_medium_evidence[start:end].add_(
                (update_mass * medium_share * observed).to(torch.float16)
            )
            self.window_surface_evidence[start:end].add_(
                (update_mass * surface_share * observed).to(torch.float16)
            )
            self.window_observation_mass[start:end].add_(
                (update_mass * observed).to(torch.float16)
            )

    @torch.no_grad()
    def update(self, **kwargs) -> None:
        self.update_from_responsibilities(**kwargs)


# Public name retained for configs and older imports.
SurfaceCarrierField = PhysicsIdentifiableDualCarrier
