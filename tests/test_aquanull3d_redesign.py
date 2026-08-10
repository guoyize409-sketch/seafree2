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
    assert torch.all(carrier.epoch_medium > carrier.epoch_surface)


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
    assert float(carrier.medium_evidence[-1]) == 1.0
    assert int(carrier.certified_state[-1]) == carrier.UNCERTAIN
    assert int(carrier.sfm_reliability[-1]) == 120

    split(params, optimizers, state, torch.tensor([True, False, False, False], device=device))
    carrier.sync_from_strategy_state(state)
    assert carrier.num_points == 5
    torch.testing.assert_close(carrier.medium_evidence[-2:], torch.ones(2, device=device))
    assert not bool(carrier.certified_state[-2:].any())

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
