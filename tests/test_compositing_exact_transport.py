import pytest
import torch

from seafree_gs.path_integrated_water import (
    CompactHeterogeneousWaterField,
    PathIntegratedWaterRenderer,
)


def test_far_gaussian_receives_stronger_transport_than_near_gaussian():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField(resolution=8).to(device)
    renderer = PathIntegratedWaterRenderer(field, query_downscale=4, depth_scale=10.0)
    means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 20.0]], device=device)
    intrinsic = torch.full((1, 2, 3), 0.4, device=device, requires_grad=True)
    radii = torch.ones(1, 2, device=device, dtype=torch.int32)
    camera_to_world = torch.eye(4, device=device).unsqueeze(0)

    transformed = renderer.transform_projected_colors(
        intrinsic, means, camera_to_world, radii
    )
    near_underwater, far_underwater = transformed[0, :, :3]
    assert transformed.shape == (1, 2, 6)
    assert not torch.allclose(near_underwater, far_underwater)
    transformed.sum().backward()
    assert intrinsic.grad is not None and torch.isfinite(intrinsic.grad).all()
    assert field.planes.grad is not None and torch.isfinite(field.planes.grad).all()


def test_training_transport_can_skip_unused_intrinsic_channels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField(resolution=8).to(device)
    renderer = PathIntegratedWaterRenderer(field, query_downscale=4, depth_scale=10.0)
    means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 20.0]], device=device)
    intrinsic = torch.full((1, 2, 3), 0.4, device=device, requires_grad=True)
    radii = torch.ones(1, 2, device=device, dtype=torch.int32)
    camera_to_world = torch.eye(4, device=device).unsqueeze(0)

    transformed = renderer.transform_projected_colors(
        intrinsic,
        means,
        camera_to_world,
        radii,
        include_intrinsic=False,
    )

    assert transformed.shape == (1, 2, 3)
    transformed.sum().backward()
    assert intrinsic.grad is not None and torch.isfinite(intrinsic.grad).all()


def test_medium_is_bounded_spectrally_ordered_and_uses_fixed_optical_scale():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field_a = CompactHeterogeneousWaterField().to(device)
    field_b = CompactHeterogeneousWaterField().to(device)
    points = torch.tensor([[-2.0, -1.0, -3.0], [2.0, 1.0, 3.0]], device=device)
    field_a.set_bounds(points)
    field_b.set_bounds(points * 7.0)
    directions = torch.nn.functional.normalize(torch.randn(12, 3, device=device), dim=-1)
    ambient, backscatter, attenuation = field_a.properties_for_directions(directions)
    assert torch.all((ambient >= 0.01) & (ambient <= 0.99))
    assert torch.all(backscatter[:, 1] >= backscatter[:, 0])
    assert torch.all(backscatter[:, 2] >= backscatter[:, 0])
    assert torch.all(attenuation[:, 0] >= attenuation[:, 1])
    assert torch.all(attenuation[:, 1] >= attenuation[:, 2])
    torch.testing.assert_close(
        field_a.normalize_distance(torch.tensor([3.0], device=device)),
        torch.tensor([0.3], device=device),
    )
    torch.testing.assert_close(
        field_b.normalize_distance(torch.tensor([21.0], device=device)),
        torch.tensor([2.1], device=device),
    )


def test_zero_initialized_field_is_exactly_identity_on_base_wpp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField().to(device)
    count = 19
    positions = torch.randn(count, 3, device=device)
    directions = torch.nn.functional.normalize(torch.randn(count, 3, device=device), dim=-1)
    ambient = torch.rand(count, 3, device=device) * 0.8 + 0.1
    backscatter = torch.rand(count, 3, device=device) + 0.05
    attenuation = torch.rand(count, 3, device=device) + 0.05
    corrected_ambient, corrected_backscatter, corrected_attenuation, residual = (
        field.apply_to_base(
            ambient, backscatter, attenuation, positions, directions
        )
    )
    assert torch.equal(corrected_ambient, ambient)
    assert torch.equal(corrected_backscatter, backscatter)
    assert torch.equal(corrected_attenuation, attenuation)
    torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_two_segment_zero_field_reduces_to_base_formula():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField(resolution=8).to(device)
    renderer = PathIntegratedWaterRenderer(field, depth_scale=10.0, far_distance=4.0)
    count = 23
    ambient = torch.rand(count, 3, device=device) * 0.7 + 0.1
    backscatter = torch.rand(count, 3, device=device) * 0.3 + 0.02
    attenuation = torch.rand(count, 3, device=device) * 0.4 + 0.02
    origins = torch.zeros(count, 3, device=device)
    directions = torch.nn.functional.normalize(torch.randn(count, 3, device=device), dim=-1)
    distances = torch.rand(count, device=device) * 25.0 + 0.1
    path = renderer.integrate_base_paths(
        ambient, backscatter, attenuation, origins, directions, distances
    )
    normalized = (distances / 10.0).clamp(0.0, 4.0)[:, None]
    torch.testing.assert_close(
        path["direct_transmittance"], torch.exp(-attenuation * normalized), rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        path["backscatter_radiance"],
        ambient * (1.0 - torch.exp(-backscatter * normalized)),
        rtol=1e-6,
        atol=1e-7,
    )
    assert field.planes.shape[1] == 4


def test_nonzero_field_is_spatially_varying_and_gateable():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField(resolution=8).to(device)
    field.set_bounds(torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device))
    with torch.no_grad():
        field.free_space_support.fill_(1.0)
        x_ramp = torch.linspace(-1.0, 1.0, 8, device=device).view(1, 1, 1, 1, 8)
        field.volume[:, 0:1].copy_(0.8 * x_ramp.expand(1, 1, 8, 8, 8))
        field.volume[:, 1:2].copy_(0.3 * x_ramp.expand(1, 1, 8, 8, 8))
        field.volume[:, 2:3].copy_(-0.2 * x_ramp.expand(1, 1, 8, 8, 8))
        field.volume[:, 3:4].copy_(0.4 * x_ramp.expand(1, 1, 8, 8, 8))
    positions = torch.tensor([[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]], device=device)
    ambient = torch.tensor([0.2, 0.4, 0.6], device=device).expand(2, -1)
    backscatter = torch.tensor([0.08, 0.16, 0.24], device=device).expand(2, -1)
    attenuation = torch.tensor([0.30, 0.18, 0.09], device=device).expand(2, -1)
    corrected_a, corrected_b, corrected_s, residual = field.apply_to_base(
        ambient, backscatter, attenuation, positions
    )
    assert not torch.allclose(residual[0], residual[1])
    assert not torch.allclose(corrected_b[0], corrected_b[1])
    assert not torch.allclose(corrected_s[0], corrected_s[1])
    assert torch.all(torch.isfinite(corrected_a))
    identity_a, identity_b, identity_s, _ = field.apply_to_base(
        ambient,
        backscatter,
        attenuation,
        positions,
        spatial_gate=torch.zeros(2, device=device),
    )
    torch.testing.assert_close(identity_a, ambient)
    torch.testing.assert_close(identity_b, backscatter)
    torch.testing.assert_close(identity_s, attenuation)


def test_default_saft_field_has_compact_parameter_budget_and_surface_support():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField().to(device)
    assert field.resolution == 16
    assert sum(parameter.numel() for parameter in field.parameters()) == 4 * 16**3
    field.set_bounds(torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device))
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [-0.9, -0.9, -0.9]], device=device
    )
    field.update_free_space_support(
        positions,
        medium_probability=torch.tensor([0.0, 1.0, 1.0], device=device),
        surface_probability=torch.tensor([1.0, 0.0, 0.0], device=device),
        ema=0.0,
    )
    support = field._sample_free_space_support(positions).squeeze(-1)
    assert float(support[0]) < float(support[1])
    assert float(support[0]) < float(support[2])


def test_zero_field_background_is_exactly_base_ambient():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField(resolution=8).to(device)
    renderer = PathIntegratedWaterRenderer(field, query_downscale=4, depth_scale=10.0)
    height, width = 24, 32
    ambient = torch.rand(height, width, 3, device=device) * 0.8 + 0.1
    backscatter = torch.rand(height, width, 3, device=device) * 0.2 + 0.02
    attenuation = torch.rand(height, width, 3, device=device) * 0.3 + 0.02
    directions = torch.nn.functional.normalize(
        torch.randn(height, width, 3, device=device), dim=-1
    )
    result = renderer.integrate_background_from_base(
        ambient,
        backscatter,
        attenuation,
        directions,
        torch.zeros(3, device=device),
        enabled=True,
    )
    torch.testing.assert_close(result["backscatter_radiance"], ambient, rtol=0.0, atol=0.0)


def test_zero_field_prefix_is_exact_base_transport_without_device_sync_metadata():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField(resolution=8).to(device)
    field.set_bounds(torch.tensor([[-2.0, -2.0, -1.0], [2.0, 2.0, 6.0]], device=device))
    renderer = PathIntegratedWaterRenderer(
        field, query_downscale=4, depth_scale=10.0, num_depth_bins=8
    )
    height, width = 32, 40
    intrinsics = torch.tensor(
        [[[50.0, 0.0, width / 2], [0.0, 50.0, height / 2], [0.0, 0.0, 1.0]]],
        device=device,
    )
    camera_to_world = torch.eye(4, device=device).unsqueeze(0)
    ambient_value = torch.tensor([0.18, 0.38, 0.48], device=device)
    backscatter_value = torch.tensor([0.08, 0.16, 0.22], device=device)
    attenuation_value = torch.tensor([0.30, 0.16, 0.08], device=device)

    def fixed_properties(directions):
        count = directions.shape[0]
        return (
            ambient_value.expand(count, -1),
            backscatter_value.expand(count, -1),
            attenuation_value.expand(count, -1),
        )

    prefix = renderer.build_frustum_prefix(
        height,
        width,
        intrinsics,
        camera_to_world,
        torch.float32,
        fixed_properties,
        enabled=True,
    )
    assert isinstance(prefix["height"], int) and isinstance(prefix["width"], int)
    means2d = torch.tensor([[20.0, 16.0], [8.0, 10.0], [31.0, 24.0]], device=device)
    depths = torch.tensor([1.5, 3.0, 5.0], device=device)
    visible_ids = torch.arange(3, device=device)
    base_ambient = ambient_value.expand(3, -1)
    base_backscatter = backscatter_value.expand(3, -1)
    base_attenuation = attenuation_value.expand(3, -1)
    dx = (means2d[:, 0] - intrinsics[0, 0, 2]) / intrinsics[0, 0, 0]
    dy = (means2d[:, 1] - intrinsics[0, 1, 2]) / intrinsics[0, 1, 1]
    ray_directions = torch.nn.functional.normalize(
        torch.stack([dx, dy, torch.ones_like(dx)], dim=-1), dim=-1
    )
    distance = depths * torch.sqrt(dx * dx + dy * dy + 1.0)
    path = renderer.sample_projected_paths(
        prefix,
        means2d,
        depths,
        visible_ids,
        base_ambient,
        base_backscatter,
        base_attenuation,
        ray_directions=ray_directions,
        path_distances=distance,
    )
    normalized = (distance / 10.0)[:, None]
    torch.testing.assert_close(
        path["direct_transmittance"],
        torch.exp(-base_attenuation * normalized),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        path["backscatter_radiance"],
        base_ambient * (1.0 - torch.exp(-base_backscatter * normalized)),
        rtol=0.0,
        atol=0.0,
    )

    background = renderer.integrate_background_from_base(
        ambient_value.expand(height, width, -1),
        backscatter_value.expand(height, width, -1),
        attenuation_value.expand(height, width, -1),
        torch.zeros(height, width, 3, device=device),
        camera_to_world[0, :3, 3],
        enabled=True,
        prefix=prefix,
    )
    assert torch.equal(
        background["backscatter_radiance"], ambient_value.expand(height, width, -1)
    )


def test_prefix_transport_backpropagates_to_compact_volume():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField(resolution=8).to(device)
    with torch.no_grad():
        field.free_space_support.fill_(1.0)
    renderer = PathIntegratedWaterRenderer(field, query_downscale=4, num_depth_bins=8)
    intrinsics = torch.tensor(
        [[[48.0, 0.0, 16.0], [0.0, 48.0, 16.0], [0.0, 0.0, 1.0]]], device=device
    )
    camera_to_world = torch.eye(4, device=device).unsqueeze(0)

    def fixed_properties(directions):
        count = directions.shape[0]
        return (
            directions.new_tensor([0.18, 0.38, 0.48]).expand(count, -1),
            directions.new_tensor([0.08, 0.16, 0.22]).expand(count, -1),
            directions.new_tensor([0.30, 0.16, 0.08]).expand(count, -1),
        )

    prefix = renderer.build_frustum_prefix(
        32, 32, intrinsics, camera_to_world, torch.float32, fixed_properties, enabled=True
    )
    means2d = torch.tensor([[16.0, 16.0], [12.0, 20.0]], device=device)
    depths = torch.tensor([1.0, 2.0], device=device)
    dx = (means2d[:, 0] - intrinsics[0, 0, 2]) / intrinsics[0, 0, 0]
    dy = (means2d[:, 1] - intrinsics[0, 1, 2]) / intrinsics[0, 1, 1]
    ray_directions = torch.nn.functional.normalize(
        torch.stack([dx, dy, torch.ones_like(dx)], dim=-1), dim=-1
    )
    distances = depths * torch.sqrt(dx * dx + dy * dy + 1.0)
    path = renderer.sample_projected_paths(
        prefix,
        means2d,
        depths,
        torch.arange(2, device=device),
        ray_directions=ray_directions,
        path_distances=distances,
    )
    (path["direct_transmittance"].sum() + path["backscatter_radiance"].sum()).backward()
    assert field.volume.grad is not None
    assert torch.isfinite(field.volume.grad).all()
    assert bool((field.volume.grad.abs() > 0).any())


def test_configured_field_scales_control_spatial_correction_strength():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weak = CompactHeterogeneousWaterField(
        resolution=8, ambient_residual_scale=0.01, coefficient_log_scale=0.05
    ).to(device)
    strong = CompactHeterogeneousWaterField(
        resolution=8, ambient_residual_scale=0.20, coefficient_log_scale=0.60
    ).to(device)
    with torch.no_grad():
        weak.volume[:, :, :, :, :4] = -0.8
        weak.volume[:, :, :, :, 4:] = 0.8
        strong.volume.copy_(weak.volume)
        weak.free_space_support.fill_(1.0)
        strong.free_space_support.fill_(1.0)
    positions = torch.tensor([[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]], device=device)
    weak_values = weak.multipliers(positions)
    strong_values = strong.multipliers(positions)
    weak_delta = sum((value - 1.0).abs().mean() for value in weak_values[:3])
    strong_delta = sum((value - 1.0).abs().mean() for value in strong_values[:3])
    assert float(strong_delta) > float(weak_delta)


def test_unsupported_field_is_identity_even_with_nonzero_latents():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField(resolution=8).to(device)
    with torch.no_grad():
        field.volume.normal_(mean=0.0, std=0.5)
    positions = torch.randn(7, 3, device=device)
    ambient = torch.rand(7, 3, device=device)
    backscatter = torch.rand(7, 3, device=device) + 0.05
    attenuation = torch.rand(7, 3, device=device) + 0.05
    corrected = field.apply_to_base(ambient, backscatter, attenuation, positions)
    assert torch.equal(corrected[0], ambient)
    assert torch.equal(corrected[1], backscatter)
    assert torch.equal(corrected[2], attenuation)


def test_freewater_observations_anchor_ambient_without_cpu_state():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField().to(device)
    image = torch.zeros(32, 32, 3, device=device)
    image[..., 1] = 0.55
    image[..., 2] = 0.70
    responsibility = torch.ones(32, 32, 1, device=device)
    before = field.ambient_anchor.clone()
    field.observe_freewater(image, responsibility, ema=0.0)
    assert not torch.allclose(field.ambient_anchor, before)
    torch.testing.assert_close(field.ambient_anchor, image[0, 0])


def test_saft_block_coordinate_schedule_separates_field_from_appearance_and_wpp():
    from types import SimpleNamespace

    from seafree_gs.seafree_model import SeaFreeGsModel

    model = object.__new__(SeaFreeGsModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        enable_path_integrated_transport=True,
        water_field_start_step=15000,
        water_field_freeze_step=23000,
        water_field_update_every=4,
    )
    model.gauss_params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(2, 3)),
            "scales": torch.nn.Parameter(torch.zeros(2, 3)),
            "quats": torch.nn.Parameter(torch.zeros(2, 4)),
            "features_dc": torch.nn.Parameter(torch.zeros(2, 3)),
            "features_rest": torch.nn.Parameter(torch.zeros(2, 3)),
            "opacities": torch.nn.Parameter(torch.zeros(2, 1)),
        }
    )
    model.line_of_sight_direction_encoding = torch.nn.Linear(3, 3)
    model.water_properties_predictor = torch.nn.Linear(3, 9)
    model.water_formation_calibrator = torch.nn.Linear(3, 9)
    model.path_integrated_renderer = PathIntegratedWaterRenderer(
        CompactHeterogeneousWaterField()
    )
    optimizers = SimpleNamespace(optimizers={})

    class RecordingStrategy:
        def __init__(self):
            self.pre_backward_calls = 0

        def step_pre_backward(self, *args, **kwargs):
            self.pre_backward_calls += 1

    model.strategy = RecordingStrategy()
    model.strategy_state = {}
    model.info = {"means2d": torch.zeros(1, 2, 2, requires_grad=False)}
    model.train()

    model.step_cb(optimizers, 15000)
    assert model.path_integrated_renderer.field.volume.requires_grad
    assert not model.gauss_params["features_dc"].requires_grad
    assert not model.gauss_params["opacities"].requires_grad
    assert not model.gauss_params["means"].requires_grad
    assert not model.gauss_params["scales"].requires_grad
    assert not model.gauss_params["quats"].requires_grad
    assert not next(model.water_properties_predictor.parameters()).requires_grad
    model._prepare_gaussian_densification_backward()
    assert model.strategy.pre_backward_calls == 0

    model.step_cb(optimizers, 15001)
    assert not model.path_integrated_renderer.field.volume.requires_grad
    assert model.gauss_params["features_dc"].requires_grad
    assert model.gauss_params["opacities"].requires_grad
    assert model.gauss_params["means"].requires_grad
    assert model.gauss_params["scales"].requires_grad
    assert model.gauss_params["quats"].requires_grad
    assert next(model.water_properties_predictor.parameters()).requires_grad
    model._prepare_gaussian_densification_backward()
    assert model.strategy.pre_backward_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterization requires CUDA")
def test_projection_transform_uses_one_raster_and_backpropagates():
    from gsplat import rasterize_responsibilities
    from gsplat.rendering import rasterization

    device = torch.device("cuda")
    field = CompactHeterogeneousWaterField(resolution=8).to(device)
    renderer = PathIntegratedWaterRenderer(field, query_downscale=4, depth_scale=10.0)
    count = 64
    means = torch.randn(count, 3, device=device) * 0.25
    means[:, 2] += 3.0
    means.requires_grad_()
    quats = torch.zeros(count, 4, device=device)
    quats[:, 0] = 1.0
    scales = torch.full((count, 3), 0.05, device=device)
    opacities = torch.full((count,), 0.25, device=device, requires_grad=True)
    colors = torch.rand(count, 3, device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    camera_to_world = torch.inverse(viewmats)
    intrinsics = torch.tensor(
        [[[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]]],
        device=device,
    )

    received_projection = {}

    def transform(**projection):
        received_projection.update(projection)
        return renderer.transform_projected_colors(
            projection["colors"],
            means,
            camera_to_world,
            projection["radii"],
            projection["camera_ids"],
            projection["gaussian_ids"],
        )

    render, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=64,
        height=64,
        packed=False,
        render_mode="RGB+ED",
        color_transform=transform,
        collect_contributions=True,
    )
    assert render.shape == (1, 64, 64, 7)
    assert alpha.shape == (1, 64, 64, 1)
    assert info["radii"].shape == (1, count)
    assert received_projection["means2d"].data_ptr() == info["means2d"].data_ptr()
    assert float(info["gs_contributions"].sum()) > 0.0
    responsibilities = torch.zeros(1, 64, 64, 3, device=device)
    responsibilities[..., 0] = 1.0
    responsibility_mass = rasterize_responsibilities(
        means2d=info["means2d"],
        conics=info["conics"],
        opacities=info["opacities"],
        responsibilities=responsibilities,
        image_width=64,
        image_height=64,
        tile_size=info["tile_size"],
        isect_offsets=info["isect_offsets"],
        flatten_ids=info["flatten_ids"],
    )
    assert responsibility_mass.shape == (1, count, 3)
    torch.testing.assert_close(
        responsibility_mass[..., 0], info["gs_contributions"], rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        responsibility_mass[..., 1:], torch.zeros_like(responsibility_mass[..., 1:])
    )
    loss = render[..., :6].mean() + alpha.mean()
    loss.backward()
    assert colors.grad is not None and torch.isfinite(colors.grad).all()
    assert opacities.grad is not None and torch.isfinite(opacities.grad).all()
    assert field.planes.grad is not None
    assert torch.isfinite(field.planes.grad).all()
