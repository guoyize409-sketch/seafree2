import torch

from gsplat.strategy.ops import duplicate, remove, split
from seafree_gs.aquanull3d import SurfaceCarrierField


def _projection_info(device: torch.device, count: int, height: int = 32, width: int = 32):
    x = torch.linspace(4.0, width - 5.0, count, device=device)
    y = torch.linspace(4.0, height - 5.0, count, device=device).flip(0)
    return {
        "means2d": torch.stack([x, y], dim=-1).unsqueeze(0),
        "radii": torch.full((1, count), 3, device=device, dtype=torch.int32),
    }


def _mass(device: torch.device, count: int, state: int, value: float = 20.0):
    mass = torch.zeros(1, count, 3, device=device)
    mass[..., state] = value
    return mass


def _observe_epochs(carrier, info, state: int, epochs: int = 2, views: int = 4):
    for step in range(epochs * views + 1):
        carrier.update_from_responsibilities(
            step=step,
            view_id=torch.tensor(step % views, device=carrier.surface_evidence.device),
            info=info,
            medium=torch.ones(32, 32, 1, device=carrier.surface_evidence.device),
            surface=torch.zeros(32, 32, 1, device=carrier.surface_evidence.device),
            responsibility_mass=_mass(carrier.surface_evidence.device, carrier.num_points, state),
        )


def test_warmup_and_zero_contribution_retain_exact_baseline_state():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carrier = SurfaceCarrierField(16, device, update_every=1, warmup_steps=4, epoch_steps=4)
    before_surface = carrier.surface_evidence.clone()
    before_medium = carrier.medium_evidence.clone()
    for step in range(5):
        carrier.update_from_responsibilities(
            step=step,
            view_id=torch.tensor(step, device=device),
            info=_projection_info(device, 16),
            medium=torch.ones(32, 32, 1, device=device),
            surface=torch.zeros(32, 32, 1, device=device),
            responsibility_mass=torch.zeros(1, 16, 3, device=device),
        )
    torch.testing.assert_close(carrier.surface_evidence, before_surface)
    torch.testing.assert_close(carrier.medium_evidence, before_medium)
    assert torch.count_nonzero(carrier.distinct_views) == 0


def test_two_epochs_certify_medium_only_for_clear_geometry():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 24
    carrier = SurfaceCarrierField(
        count,
        device,
        update_every=1,
        min_distinct_views=4,
        warmup_steps=0,
        epoch_steps=4,
        min_certified_epochs=2,
    )
    info = _projection_info(device, count)
    _observe_epochs(carrier, info, state=0)
    weights = carrier.render_weights()
    assert bool(weights["freewater_hard"].all())
    torch.testing.assert_close(weights["clear_scene"], torch.zeros(count, device=device))
    torch.testing.assert_close(weights["underwater_geometry"], torch.ones(count, device=device))


def test_surface_and_uncertain_points_retain_both_geometries():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 20
    carrier = SurfaceCarrierField(
        count,
        device,
        update_every=1,
        min_distinct_views=4,
        warmup_steps=0,
        epoch_steps=4,
    )
    info = _projection_info(device, count)
    initial = carrier.render_weights()
    assert not bool(initial["certified"].any())
    torch.testing.assert_close(initial["clear_scene"], torch.ones(count, device=device))
    _observe_epochs(carrier, info, state=1)
    weights = carrier.render_weights()
    assert bool(weights["certified_surface"].all())
    torch.testing.assert_close(weights["clear_scene"], torch.ones(count, device=device))
    torch.testing.assert_close(weights["underwater_geometry"], torch.ones(count, device=device))


def test_exact_mass_and_repeated_view_accounting():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 12
    carrier = SurfaceCarrierField(
        count, device, update_every=1, min_distinct_views=2, warmup_steps=0, epoch_steps=20
    )
    kwargs = dict(
        info=_projection_info(device, count),
        medium=torch.ones(32, 32, 1, device=device),
        surface=torch.zeros(32, 32, 1, device=device),
        responsibility_mass=_mass(device, count, 0),
    )
    carrier.update_from_responsibilities(step=0, view_id=torch.tensor(7, device=device), **kwargs)
    carrier.update_from_responsibilities(step=1, view_id=torch.tensor(7, device=device), **kwargs)
    assert int(carrier.distinct_views.max()) == 1
    assert torch.all(carrier.window_medium_evidence > carrier.window_surface_evidence)
    torch.testing.assert_close(carrier.medium_evidence, carrier.surface_evidence)
    carrier.update_from_responsibilities(step=2, view_id=torch.tensor(8, device=device), **kwargs)
    carrier.update_from_responsibilities(step=20, view_id=torch.tensor(9, device=device), **kwargs)
    assert torch.all(carrier.medium_evidence > carrier.surface_evidence)


def test_real_view_ids_are_deduplicated_and_parallax_is_recorded():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carrier = SurfaceCarrierField(
        1,
        device,
        update_every=1,
        min_distinct_views=2,
        warmup_steps=0,
        epoch_steps=20,
    )
    info = _projection_info(device, 1)
    means = torch.zeros(1, 3, device=device)
    camera_origins = (
        torch.tensor([-1.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, -1.0, 0.0], device=device),
        torch.tensor([1.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )
    for step, camera_origin in enumerate(camera_origins):
        carrier.update_from_responsibilities(
            step=step,
            view_id=torch.tensor(3, device=device),
            info=info,
            responsibility_mass=_mass(device, 1, 0),
            means=means,
            camera_origin=camera_origin,
        )
    assert int(carrier.angular_coverage[0]) == 1

    distinct = SurfaceCarrierField(
        1,
        device,
        update_every=1,
        min_distinct_views=2,
        warmup_steps=0,
        epoch_steps=20,
    )
    for step in range(4):
        distinct.update_from_responsibilities(
            step=step,
            view_id=torch.tensor(step, device=device),
            info=info,
            responsibility_mass=_mass(device, 1, 0),
            means=means,
            camera_origin=camera_origins[step],
        )
    assert int(distinct.angular_coverage[0]) == 4
    assert float(distinct.parallax_evidence[0]) > 0.0


def test_continuous_carrier_split_is_conservative_and_exact_at_certification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carrier = SurfaceCarrierField(3, device, warmup_steps=0, min_distinct_views=2)
    carrier.medium_evidence.copy_(torch.tensor([0.55, 0.50, 0.95], device=device))
    carrier.surface_evidence.copy_(torch.tensor([0.45, 0.50, 0.05], device=device))
    carrier.observation_mass.fill_(4.0)
    carrier.lifetime_coverage.fill_(3)
    weights = carrier.render_weights()
    torch.testing.assert_close(
        weights["surface_fraction"] + weights["medium_fraction"],
        torch.ones(3, device=device),
        rtol=0.0,
        atol=0.0,
    )
    assert 0.0 < float(weights["medium_fraction"][0]) < 1.0
    assert 0.0 < float(weights["surface_fraction"][0]) < 1.0

    carrier.certified_state[2] = carrier.MEDIUM
    certified = carrier.render_weights()
    assert float(certified["clear_scene"][2]) == 0.0
    assert float(certified["medium_fraction"][2]) == 1.0
    assert float(certified["underwater_geometry"][2]) == 1.0


def test_contradictory_epoch_returns_to_uncertain_before_recertification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 8
    carrier = SurfaceCarrierField(
        count,
        device,
        update_every=1,
        min_distinct_views=4,
        warmup_steps=0,
        epoch_steps=4,
    )
    info = _projection_info(device, count)
    _observe_epochs(carrier, info, state=0)
    assert bool(carrier.render_weights()["certified_medium"].all())
    for step in range(9, 13):
        carrier.update_from_responsibilities(
            step=step,
            view_id=torch.tensor(step % 4, device=device),
            info=info,
            medium=torch.zeros(32, 32, 1, device=device),
            surface=torch.ones(32, 32, 1, device=device),
            responsibility_mass=_mass(device, count, 1),
        )
    assert not bool(carrier.render_weights()["certified"].any())
    for step in range(13, 17):
        carrier.update_from_responsibilities(
            step=step,
            view_id=torch.tensor(step % 4, device=device),
            info=info,
            medium=torch.zeros(32, 32, 1, device=device),
            surface=torch.ones(32, 32, 1, device=device),
            responsibility_mass=_mass(device, count, 1),
        )
    assert bool(carrier.render_weights()["certified_surface"].all())


def test_unobserved_updates_do_not_erase_cross_window_candidate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carrier = SurfaceCarrierField(
        2,
        device,
        update_every=1,
        min_distinct_views=3,
        warmup_steps=0,
        epoch_steps=5,
    )
    info = _projection_info(device, 2)
    for step in range(11):
        mass = torch.zeros(1, 2, 3, device=device)
        if step % 5 in (0, 2, 4):
            mass[0, 0, 0] = 20.0
        carrier.update_from_responsibilities(
            step=step,
            view_id=torch.tensor(step, device=device),
            info=info,
            responsibility_mass=mass,
        )
    assert bool(carrier.render_weights()["certified_medium"][0])
    assert not bool(carrier.render_weights()["certified"][1])


def test_geometric_counterevidence_preserves_real_surface_with_water_colored_pixels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 6
    carrier = SurfaceCarrierField(
        count,
        device,
        update_every=1,
        min_distinct_views=3,
        warmup_steps=0,
        epoch_steps=4,
    )
    info = _projection_info(device, count)
    for step in range(9):
        carrier.update_from_responsibilities(
            step=step,
            view_id=torch.tensor(step % 4, device=device),
            info=info,
            responsibility_mass=_mass(device, count, 0),
            geometric_medium_support=torch.zeros(count, device=device),
            geometric_surface_support=torch.ones(count, device=device),
        )
    assert bool(carrier.render_weights()["certified_surface"].all())
    assert not bool(carrier.render_weights()["certified_medium"].any())


def test_projection_geometry_support_separates_foreground_from_empty_water():
    from types import SimpleNamespace

    from seafree_gs.seafree_model import SeaFreeGsModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = object.__new__(SeaFreeGsModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(water_null_depth_threshold=1e-2)
    model.info = {
        "means2d": torch.tensor([[[4.0, 4.0], [12.0, 12.0]]], device=device),
        "depths": torch.tensor([[2.0, 5.0]], device=device),
        "radii": torch.ones(1, 2, device=device, dtype=torch.int32),
    }
    pseudo = torch.zeros(16, 16, 1, device=device)
    pseudo[4, 4] = 1.0
    rendered_depth = torch.full_like(pseudo, 5.0)
    rendered_depth[4, 4] = 2.0
    edge = torch.zeros_like(pseudo)
    edge[4, 4] = 1.0
    low_structure = torch.ones_like(pseudo)
    water_advantage = torch.ones_like(pseudo)
    medium, surface = model._gaussian_geometric_counterfactual_support(
        pseudo,
        rendered_depth,
        edge,
        low_structure,
        water_advantage,
    )
    assert medium is not None and surface is not None
    assert float(surface[0]) > float(medium[0])
    assert float(medium[1]) > float(surface[1])


def test_duplicate_and_split_children_require_new_evidence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 3
    carrier = SurfaceCarrierField(count, device, update_every=1, warmup_steps=0)
    carrier.medium_evidence.copy_(torch.tensor([2.0, 3.0, 4.0], device=device))
    carrier.certified_state.fill_(carrier.MEDIUM)
    carrier.sfm_reliability.copy_(torch.tensor([200, 160, 120], device=device, dtype=torch.uint8))
    state = {}
    carrier.bind_strategy_state(state)
    params = {
        "means": torch.nn.Parameter(torch.randn(count, 3, device=device), requires_grad=False),
        "scales": torch.nn.Parameter(torch.zeros(count, 3, device=device), requires_grad=False),
        "quats": torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(count, 1),
            requires_grad=False,
        ),
        "opacities": torch.nn.Parameter(torch.zeros(count, 1, device=device), requires_grad=False),
    }
    optimizers = {}
    duplicate(params, optimizers, state, torch.tensor([False, True, False], device=device))
    carrier.sync_from_strategy_state(state)
    assert carrier.num_points == 4
    expected_duplicate_evidence = 0.5 + 0.35 * (3.0 - 0.5)
    torch.testing.assert_close(
        carrier.medium_evidence[-1], torch.tensor(expected_duplicate_evidence, device=device)
    )
    assert int(carrier.certified_state[-1]) == carrier.UNCERTAIN
    assert int(carrier.sfm_reliability[-1]) == round(160 * 0.85)

    split(params, optimizers, state, torch.tensor([True, False, False, False], device=device))
    carrier.sync_from_strategy_state(state)
    assert carrier.num_points == 5
    expected_split_evidence = 0.5 + 0.35 * (2.0 - 0.5)
    torch.testing.assert_close(
        carrier.medium_evidence[-2:],
        torch.full((2,), expected_split_evidence, device=device),
    )
    assert not bool(carrier.certified_state[-2:].any())
    assert not bool(carrier.angular_coverage[-2:].any())

    remove(params, optimizers, state, torch.tensor([False, True, False, False, False], device=device))
    carrier.sync_from_strategy_state(state)
    assert carrier.num_points == 4
    assert all(value.shape[0] == 4 for value in state.values())


def test_strategy_state_rebinds_after_module_device_move():
    if not torch.cuda.is_available():
        return
    carrier = SurfaceCarrierField(8, torch.device("cpu"), update_every=1)
    state = {}
    carrier.bind_strategy_state(state)
    carrier.cuda()
    assert state["carrier_surface_evidence"].device.type == "cpu"
    carrier.sync_from_strategy_state(state)
    assert state["carrier_surface_evidence"].device.type == "cuda"
    assert carrier.surface_evidence.data_ptr() == state["carrier_surface_evidence"].data_ptr()


def test_full_view_ids_do_not_alias_above_sixty_three():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carrier = SurfaceCarrierField(
        3, device, update_every=1, min_distinct_views=2, warmup_steps=0, epoch_steps=20
    )
    info = _projection_info(device, 3)
    for step, view_id in enumerate((1, 64)):
        carrier.update_from_responsibilities(
            step=step,
            view_id=torch.tensor(view_id, device=device),
            info=info,
            responsibility_mass=_mass(device, 3, 0),
        )
    assert int(carrier.angular_coverage.min()) == 2


def test_pending_window_affects_render_and_can_be_committed_once():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carrier = SurfaceCarrierField(
        4, device, update_every=1, min_distinct_views=2, warmup_steps=0, epoch_steps=100
    )
    info = _projection_info(device, 4)
    for step in range(3):
        carrier.update_from_responsibilities(
            step=step,
            view_id=torch.tensor(step, device=device),
            info=info,
            responsibility_mass=_mass(device, 4, 0),
        )
    assert bool((carrier.render_weights()["medium_fraction"] > 0.0).all())
    before = carrier.medium_evidence.clone()
    assert carrier.finalize_pending_window()
    assert bool((carrier.medium_evidence > before).all())
    assert not carrier.finalize_pending_window()


def test_surface_medium_split_exactly_conserves_optical_thickness():
    opacity = torch.tensor([0.0, 0.1, 0.5, 0.95])
    surface_fraction = torch.tensor([0.0, 0.25, 0.7, 1.0])
    surface, medium = SurfaceCarrierField.split_optical_thickness(
        opacity, surface_fraction
    )
    recombined = 1.0 - (1.0 - surface) * (1.0 - medium)
    torch.testing.assert_close(recombined, opacity, rtol=1e-6, atol=1e-7)
    assert float(surface[0]) == 0.0 and float(medium[0]) == 0.0
    assert float(medium[-1]) == 0.0


def test_mixed_realistic_evidence_produces_nonzero_medium_ownership():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 40
    carrier = SurfaceCarrierField(
        count,
        device,
        update_every=1,
        min_distinct_views=4,
        warmup_steps=0,
        epoch_steps=8,
    )
    info = _projection_info(device, count)
    mass = torch.zeros(1, count, 3, device=device)
    mass[:, :16, 0] = 8.0
    mass[:, :16, 1] = 3.0
    mass[:, 16:, 0] = 2.0
    mass[:, 16:, 1] = 9.0
    for step in range(7):
        carrier.update_from_responsibilities(
            step=step,
            view_id=torch.tensor(step, device=device),
            info=info,
            responsibility_mass=mass,
        )
    fractions = carrier.render_weights()["medium_fraction"]
    assert float(fractions[:16].mean()) > 0.45
    assert float(fractions[16:].mean()) < 0.05


def test_high_opacity_alone_cannot_change_carrier_ownership():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 8
    low = SurfaceCarrierField(
        count, device, update_every=1, min_distinct_views=3, warmup_steps=0, epoch_steps=4
    )
    high = SurfaceCarrierField(
        count, device, update_every=1, min_distinct_views=3, warmup_steps=0, epoch_steps=4
    )
    low_info = _projection_info(device, count)
    high_info = _projection_info(device, count)
    low_info["opacities"] = torch.full((1, count), 0.01, device=device)
    high_info["opacities"] = torch.full((1, count), 0.99, device=device)
    for step in range(5):
        kwargs = {
            "step": step,
            "view_id": torch.tensor(step, device=device),
            "responsibility_mass": _mass(device, count, 0),
        }
        low.update_from_responsibilities(info=low_info, **kwargs)
        high.update_from_responsibilities(info=high_info, **kwargs)
    torch.testing.assert_close(
        low.render_weights()["medium_fraction"],
        high.render_weights()["medium_fraction"],
        rtol=0.0,
        atol=0.0,
    )
