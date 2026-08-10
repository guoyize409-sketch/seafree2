"""Persistent multi-view surface/medium ownership for underwater Gaussians."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


class RayConsistentDualCarrier(nn.Module):
    """Fuse exact raster responsibilities into conservative 3D ownership.

    The carrier deliberately separates geometry from water radiance.  Attached
    haze is not a geometry class: a Gaussian is either a supported surface, a
    certified medium carrier, or remains uncertain.  Uncertain points retain
    the baseline geometry in both renders.
    """

    UNCERTAIN = 0
    SURFACE = 1
    MEDIUM = 2

    STATE_KEYS = (
        "carrier_medium_evidence",
        "carrier_surface_evidence",
        "carrier_observation_mass",
        "carrier_epoch_medium",
        "carrier_epoch_surface",
        "carrier_epoch_uncertainty",
        "carrier_distinct_views",
        "carrier_view_mask_a",
        "carrier_view_mask_b",
        "carrier_epoch_vote",
        "carrier_certification_streak",
        "carrier_certified_state",
        "carrier_sfm_reliability",
    )

    def __init__(
        self,
        num_points: int,
        device: torch.device,
        update_every: int = 32,
        min_distinct_views: int = 6,
        num_train_views: int = 1,
        warmup_steps: int = 4000,
        epoch_steps: int = 0,
        min_certified_epochs: int = 2,
        depth_scale: float = 10.0,
        chunk_size: int = 262144,
        evidence_decay: float = 0.985,
        min_observation_mass: float = 1.5,
        sfm_reliability: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.update_every = max(int(update_every), 1)
        self.min_distinct_views = max(int(min_distinct_views), 1)
        self.num_train_views = max(int(num_train_views), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        inferred_epoch = self.num_train_views * self.update_every
        self.epoch_steps = max(
            int(epoch_steps) if epoch_steps > 0 else min(max(inferred_epoch, 2000), 6000),
            self.update_every,
        )
        self.min_certified_epochs = max(int(min_certified_epochs), 2)
        self.depth_scale = max(float(depth_scale), 1e-4)
        self.chunk_size = max(int(chunk_size), 16384)
        self.evidence_decay = min(max(float(evidence_decay), 0.90), 0.9999)
        self.min_observation_mass = max(float(min_observation_mass), 1e-4)
        self._epoch_index_value = -1
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
                sfm_reliability.detach().to(device=device, dtype=torch.float32)
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
            )
        sfm_prior = sfm_u8.float() / 255.0
        self.register_buffer("medium_evidence", torch.ones(num_points, device=device))
        self.register_buffer(
            "surface_evidence", 1.0 + 1.5 * sfm_prior.to(device=device)
        )
        self.register_buffer("observation_mass", torch.zeros(num_points, device=device))
        self.register_buffer("epoch_medium", torch.zeros(num_points, device=device, dtype=torch.float16))
        self.register_buffer("epoch_surface", torch.zeros(num_points, device=device, dtype=torch.float16))
        self.register_buffer("epoch_uncertainty", torch.zeros(num_points, device=device, dtype=torch.float16))
        self.register_buffer("distinct_views", torch.zeros(num_points, device=device, dtype=torch.uint8))
        self.register_buffer("view_mask_a", torch.zeros(num_points, device=device, dtype=torch.int32))
        self.register_buffer("view_mask_b", torch.zeros(num_points, device=device, dtype=torch.int32))
        self.register_buffer("epoch_vote", torch.zeros(num_points, device=device, dtype=torch.uint8))
        self.register_buffer("certification_streak", torch.zeros(num_points, device=device, dtype=torch.uint8))
        self.register_buffer("certified_state", torch.zeros(num_points, device=device, dtype=torch.uint8))
        self.register_buffer("sfm_reliability", sfm_u8)
        self.register_buffer("epoch_index", torch.full((), -1, device=device, dtype=torch.int32))

    @property
    def num_points(self) -> int:
        return int(self.surface_evidence.shape[0])

    # Compatibility aliases used by metrics and old checkpoints.
    @property
    def freewater_evidence(self) -> torch.Tensor:
        return self.medium_evidence

    @property
    def scene_evidence(self) -> torch.Tensor:
        return self.surface_evidence

    @property
    def haze_evidence(self) -> torch.Tensor:
        return self.epoch_uncertainty.float()

    @property
    def empty_evidence(self) -> torch.Tensor:
        return self.medium_evidence

    @property
    def water_evidence(self) -> torch.Tensor:
        return self.medium_evidence

    def resize(self, num_points: int, device: torch.device) -> None:
        if self.num_points == num_points and self.surface_evidence.device == device:
            return
        # Loading an incompatible checkpoint must be conservative: all points
        # become uncertain rather than inheriting stale ownership.
        sfm = None
        if self.sfm_reliability.numel() == num_points:
            sfm = self.sfm_reliability.float() / 255.0
        values = RayConsistentDualCarrier(
            num_points=num_points,
            device=device,
            update_every=self.update_every,
            min_distinct_views=self.min_distinct_views,
            num_train_views=self.num_train_views,
            warmup_steps=self.warmup_steps,
            epoch_steps=self.epoch_steps,
            min_certified_epochs=self.min_certified_epochs,
            depth_scale=self.depth_scale,
            chunk_size=self.chunk_size,
            evidence_decay=self.evidence_decay,
            min_observation_mass=self.min_observation_mass,
            sfm_reliability=sfm,
        )
        for name, value in values.named_buffers():
            setattr(self, name, value)
        self._epoch_index_value = -1

    def _state_values(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.medium_evidence,
            self.surface_evidence,
            self.observation_mass,
            self.epoch_medium,
            self.epoch_surface,
            self.epoch_uncertainty,
            self.distinct_views,
            self.view_mask_a,
            self.view_mask_b,
            self.epoch_vote,
            self.certification_streak,
            self.certified_state,
            self.sfm_reliability,
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
            self.epoch_medium,
            self.epoch_surface,
            self.epoch_uncertainty,
            self.distinct_views,
            self.view_mask_a,
            self.view_mask_b,
            self.epoch_vote,
            self.certification_streak,
            self.certified_state,
            self.sfm_reliability,
        ) = typed

    @torch.no_grad()
    def _finalize_epoch(self) -> None:
        medium = self.epoch_medium.float()
        surface = self.epoch_surface.float()
        uncertain = self.epoch_uncertainty.float()
        decisive = medium + surface
        total = decisive + uncertain
        medium_probability = medium / decisive.clamp_min(1e-5)
        uncertainty_fraction = uncertain / total.clamp_min(1e-5)
        enough_views = self.distinct_views >= self.min_distinct_views
        enough_mass = decisive >= self.min_observation_mass
        sfm = self.sfm_reliability.float() / 255.0

        medium_vote = (
            enough_views
            & enough_mass
            & (medium_probability > 0.64)
            & (uncertainty_fraction < 0.38)
            & (sfm < 0.55)
        )
        surface_vote = (
            enough_views
            & enough_mass
            & ((medium_probability < 0.43) | (sfm > 0.72))
        )
        vote = torch.zeros_like(self.epoch_vote)
        vote[surface_vote] = self.SURFACE
        vote[medium_vote & ~surface_vote] = self.MEDIUM

        same = (vote == self.epoch_vote) & (vote != self.UNCERTAIN)
        next_streak = torch.where(
            same,
            (self.certification_streak.to(torch.int16) + 1).clamp_max(255).to(torch.uint8),
            (vote != self.UNCERTAIN).to(torch.uint8),
        )
        committed = next_streak >= self.min_certified_epochs
        # Contradictory or ambiguous epochs immediately fall back to baseline.
        next_state = torch.where(committed, vote, torch.zeros_like(vote))
        self.certified_state.copy_(next_state)
        self.certification_streak.copy_(next_streak)
        self.epoch_vote.copy_(vote)

        epoch_scale = total.clamp(0.0, 8.0)
        keep = self.evidence_decay
        self.medium_evidence.mul_(keep).add_((1.0 - keep) * epoch_scale * medium_probability)
        self.surface_evidence.mul_(keep).add_((1.0 - keep) * epoch_scale * (1.0 - medium_probability))
        self.epoch_medium.zero_()
        self.epoch_surface.zero_()
        self.epoch_uncertainty.zero_()
        self.distinct_views.zero_()
        self.view_mask_a.zero_()
        self.view_mask_b.zero_()

    @torch.no_grad()
    def _roll_epoch(self, step: int) -> None:
        if step < self.warmup_steps:
            return
        epoch = (step - self.warmup_steps) // self.epoch_steps
        previous = self._epoch_index_value
        if previous < 0:
            self.epoch_index.fill_(epoch)
            self._epoch_index_value = epoch
        elif epoch > previous:
            self._finalize_epoch()
            self.epoch_index.fill_(epoch)
            self._epoch_index_value = epoch

    def probabilities(self) -> Dict[str, torch.Tensor]:
        medium = self.medium_evidence.clamp_min(1e-5)
        surface = self.surface_evidence.clamp_min(1e-5)
        probability_medium = medium / (medium + surface)
        probability_surface = 1.0 - probability_medium
        geometry = torch.stack([probability_medium, probability_surface], dim=-1).clamp_min(1e-5)
        entropy = -(geometry * geometry.log()).sum(dim=-1) / geometry.new_tensor(2.0).log()
        observation_uncertainty = torch.exp(-self.observation_mass / self.min_observation_mass)
        uncertainty = torch.maximum(entropy, observation_uncertainty).clamp(0.0, 1.0)
        certified = self.certified_state != self.UNCERTAIN
        observed = certified | (
            (self.distinct_views >= self.min_distinct_views)
            & (self.observation_mass >= self.min_observation_mass)
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
        medium_hard = ownership["certified_medium"]
        surface_gate = (~medium_hard).to(self.surface_evidence)
        # Carrier decisions never alter underwater geometry.  The transport
        # field may explain certified medium radiance without destructive prune.
        underwater_gate = torch.ones_like(surface_gate)
        return {
            **ownership,
            "clear_scene": surface_gate,
            "underwater_geometry": underwater_gate,
            "surface_geometry": surface_gate,
            "transport_confidence": medium_hard.to(surface_gate),
            "freewater_hard": medium_hard,
            "haze_hard": torch.zeros_like(medium_hard),
        }

    def probability(self) -> torch.Tensor:
        return self.render_weights()["clear_scene"]

    def lifecycle_masks(
        self,
        step: int,
        grow_start: int,
        prune_start: int,
        max_prune_fraction: float,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        del prune_start, max_prune_fraction
        hard = self.render_weights()["freewater_hard"]
        # Suppressing future duplication is reversible; pruning is not.
        return (hard if step >= grow_start else None), None

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
        if medium is None:
            medium = freewater
        if surface is None:
            surface = scene
        if medium is None or surface is None:
            return None
        if uncertainty is None:
            uncertainty = haze if haze is not None else (1.0 - torch.maximum(medium, surface))
        responsibilities = torch.cat([medium, surface, uncertainty], dim=-1)
        responsibilities = responsibilities.detach().float().clamp(0.0, 1.0)
        if valid_image_mask is not None:
            responsibilities *= (valid_image_mask[..., :1].detach().to(responsibilities) > 0.5)

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
    ) -> None:
        del contribution_mass, valid_image_mask, haze
        if step < self.warmup_steps or step % self.update_every != 0:
            return
        self._roll_epoch(step)
        if responsibility_mass is None or responsibility_mass.numel() == 0:
            return
        if medium is None:
            medium = freewater
        if surface is None:
            surface = scene
        if medium is None or surface is None:
            return

        means2d = info.get("means2d")
        radii = info.get("radii")
        if means2d is None or radii is None or means2d.dim() != 3 or means2d.shape[0] != 1:
            return
        exact = responsibility_mass.reshape(-1, 3).detach().float().clamp_min(0.0)
        n = min(self.num_points, means2d.shape[1], exact.shape[0])
        if n == 0:
            return

        current_view = view_id.to(means2d.device, torch.int64).reshape(-1)[0]
        slot = torch.remainder(current_view, 62)
        bank = torch.div(slot, 31, rounding_mode="floor")
        shift = torch.remainder(slot, 31).to(torch.int32)
        bit = torch.bitwise_left_shift(
            torch.ones((), device=means2d.device, dtype=torch.int32), shift
        )

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            weighted = exact[start:end]
            contribution = weighted.sum(dim=-1)
            observed_contribution = 1.0 - torch.exp(-contribution / 4.0)
            response = weighted / contribution[:, None].clamp_min(1e-5)
            masks = (self.view_mask_a[start:end], self.view_mask_b[start:end])
            seen = torch.zeros_like(observed_contribution, dtype=torch.bool)
            for bank_id, mask in enumerate(masks):
                seen = torch.where(bank == bank_id, torch.bitwise_and(mask, bit) != 0, seen)
            new_view = (observed_contribution > 1e-3) & ~seen
            update_mass = observed_contribution * new_view.float()

            self.epoch_medium[start:end].add_((update_mass * response[:, 0]).to(torch.float16))
            self.epoch_surface[start:end].add_((update_mass * response[:, 1]).to(torch.float16))
            self.epoch_uncertainty[start:end].add_((update_mass * response[:, 2]).to(torch.float16))
            self.observation_mass[start:end].add_(update_mass).clamp_(max=255.0)
            count = self.distinct_views[start:end].to(torch.int16) + new_view.to(torch.int16)
            self.distinct_views[start:end].copy_(count.clamp_max(255).to(torch.uint8))
            add_bit = torch.where(
                new_view,
                bit.expand_as(self.view_mask_a[start:end]),
                torch.zeros_like(self.view_mask_a[start:end]),
            )
            for bank_id, mask in enumerate(masks):
                mask.copy_(torch.where(bank == bank_id, torch.bitwise_or(mask, add_bit), mask))

    @torch.no_grad()
    def update(self, **kwargs) -> None:
        self.update_from_responsibilities(**kwargs)


# Keep the public name used by configs and older imports.
SurfaceCarrierField = RayConsistentDualCarrier
