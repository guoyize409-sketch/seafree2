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
    assert field.planes.shape[1] == 3


def test_nonzero_field_uses_shared_rgb_multipliers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = CompactHeterogeneousWaterField(resolution=8).to(device)
    with torch.no_grad():
        field.planes[:, 0].fill_(0.7)
        field.planes[:, 1].fill_(0.5)
        field.planes[:, 2].fill_(-0.4)
    count = 11
    positions = torch.randn(count, 3, device=device)
    ambient = torch.tensor([0.2, 0.4, 0.6], device=device).expand(count, -1)
    backscatter = torch.tensor([0.08, 0.16, 0.24], device=device).expand(count, -1)
    attenuation = torch.tensor([0.30, 0.18, 0.09], device=device).expand(count, -1)
    corrected_a, corrected_b, corrected_s, _ = field.apply_to_base(
        ambient, backscatter, attenuation, positions
    )
    torch.testing.assert_close(
        corrected_b / backscatter,
        (corrected_b / backscatter)[..., :1].expand_as(corrected_b),
    )
    torch.testing.assert_close(
        corrected_s / attenuation,
        (corrected_s / attenuation)[..., :1].expand_as(corrected_s),
    )
    assert torch.all(corrected_a[:, 2] > corrected_a[:, 1])
    assert torch.all(corrected_b[:, 2] > corrected_b[:, 1])
    assert torch.all(corrected_s[:, 0] > corrected_s[:, 1])


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

    def transform(**projection):
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
