#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,   // [C, N] or [nnz]
    const S *__restrict__ backgrounds, // [C, COLOR_DIM]
    const bool *__restrict__ masks,    // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    S *__restrict__ render_colors, // [C, image_height, image_width, COLOR_DIM]
    S *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    S *__restrict__ pixels, // [C, N] or [nnz]
    const bool collect_contributions
    
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_colors += camera_id * image_height * image_width * COLOR_DIM;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    S px = (S)j + 0.5f;
    S py = (S)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]
        ); // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    S T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    S pix_out[COLOR_DIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<S> conic = conic_batch[t];
            const vec3<S> xy_opac = xy_opacity_batch[t];
            const S opac = xy_opac.z;
            const vec2<S> delta = {xy_opac.x - px, xy_opac.y - py};
            const S sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
            S alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const S next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const S vis = alpha * T;
            const S *c_ptr = colors + g * COLOR_DIM;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            cur_idx = batch_start + t;

            T = next_T;
            if (collect_contributions) {
                gpuAtomicAdd(pixels + g, vis);
            }
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        render_alphas[pix_id] = 1.0f - T;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    const bool collect_contributions
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t channels = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor renders = torch::empty(
        {C, image_height, image_width, channels},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor alphas = torch::empty(
        {C, image_height, image_width, 1},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor last_ids = torch::empty(
        {C, image_height, image_width}, means2d.options().dtype(torch::kInt32)
    );
    torch::Tensor pixels = torch::zeros_like(opacities);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }
    rasterize_to_pixels_fwd_kernel<CDIM, float>
        <<<blocks, threads, shared_mem, stream>>>(
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            pixels.data_ptr<float>(),
            collect_contributions
        );

    return std::make_tuple(renders, alphas, last_ids, pixels);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    const bool collect_contributions
) {
    GSPLAT_CHECK_INPUT(colors);
    uint32_t channels = colors.size(-1);

#define __GS__CALL_(N)                                                         \
    case N:                                                                    \
        return call_kernel_with_dim<N>(                                        \
            means2d,                                                           \
            conics,                                                            \
            colors,                                                            \
            opacities,                                                         \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            collect_contributions                                              \
        );

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    switch (channels) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
}

/****************************************************************************
 * No-grad Pixel Responsibility Reduction
 ****************************************************************************/

__global__ void rasterize_responsibilities_kernel(
    const uint32_t C,
    const uint32_t n_isects,
    const vec2<float> *__restrict__ means2d,
    const vec3<float> *__restrict__ conics,
    const float *__restrict__ opacities,
    const float *__restrict__ responsibilities,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    float *__restrict__ gaussian_mass
) {
    auto block = cg::this_thread_block();
    const int32_t camera_id = block.group_index().x;
    const int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i =
        block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j =
        block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    responsibilities += camera_id * image_height * image_width * 3;

    const float px = static_cast<float>(j) + 0.5f;
    const float py = static_cast<float>(i) + 0.5f;
    const int32_t pix_id = i * image_width + j;
    const bool inside = i < image_height && j < image_width;
    bool done = !inside;

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (camera_id == C - 1) &&
                (tile_id == static_cast<int32_t>(tile_width * tile_height - 1))
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int shared[];
    int32_t *id_batch = reinterpret_cast<int32_t *>(shared);
    vec3<float> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]);
    vec3<float> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]);

    float T = 1.0f;
    const uint32_t thread_rank = block.thread_rank();
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        if (__syncthreads_count(done) >= block_size) {
            break;
        }
        const uint32_t batch_start = range_start + block_size * batch;
        const uint32_t idx = batch_start + thread_rank;
        if (idx < static_cast<uint32_t>(range_end)) {
            const int32_t gaussian_id = flatten_ids[idx];
            id_batch[thread_rank] = gaussian_id;
            const vec2<float> xy = means2d[gaussian_id];
            xy_opacity_batch[thread_rank] = {
                xy.x, xy.y, opacities[gaussian_id]
            };
            conic_batch[thread_rank] = conics[gaussian_id];
        }
        block.sync();

        const uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t item = 0; item < batch_size && !done; ++item) {
            const vec3<float> conic = conic_batch[item];
            const vec3<float> xy_opacity = xy_opacity_batch[item];
            const vec2<float> delta = {xy_opacity.x - px, xy_opacity.y - py};
            const float sigma =
                0.5f * (conic.x * delta.x * delta.x +
                        conic.z * delta.y * delta.y) +
                conic.y * delta.x * delta.y;
            const float alpha =
                min(0.999f, xy_opacity.z * __expf(-sigma));
            if (sigma < 0.0f || alpha < 1.0f / 255.0f) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }

            const float r0 = max(responsibilities[pix_id * 3], 0.0f);
            const float r1 = max(responsibilities[pix_id * 3 + 1], 0.0f);
            const float r2 = max(responsibilities[pix_id * 3 + 2], 0.0f);
            const float responsibility_sum = r0 + r1 + r2;
            if (responsibility_sum > 1e-6f) {
                const float visibility = alpha * T / responsibility_sum;
                const int32_t gaussian_id = id_batch[item];
                gpuAtomicAdd(gaussian_mass + gaussian_id * 3, visibility * r0);
                gpuAtomicAdd(gaussian_mass + gaussian_id * 3 + 1, visibility * r1);
                gpuAtomicAdd(gaussian_mass + gaussian_id * 3 + 2, visibility * r2);
            }
            T = next_T;
        }
    }
}

torch::Tensor rasterize_responsibilities_tensor(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &responsibilities,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const torch::Tensor &tile_offsets,
    const torch::Tensor &flatten_ids
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(responsibilities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    TORCH_CHECK(means2d.scalar_type() == torch::kFloat32, "means2d must be float32");
    TORCH_CHECK(conics.scalar_type() == torch::kFloat32, "conics must be float32");
    TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
    TORCH_CHECK(
        responsibilities.scalar_type() == torch::kFloat32,
        "responsibilities must be float32"
    );
    TORCH_CHECK(
        responsibilities.dim() == 4 && responsibilities.size(3) == 3,
        "responsibilities must have shape [C,H,W,3]"
    );

    const uint32_t C = tile_offsets.size(0);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width = tile_offsets.size(2);
    const uint32_t n_isects = flatten_ids.size(0);
    torch::Tensor gaussian_mass = torch::zeros(
        {opacities.numel(), 3}, opacities.options().dtype(torch::kFloat32)
    );

    const dim3 threads = {tile_size, tile_size, 1};
    const dim3 blocks = {C, tile_height, tile_width};
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>));
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    rasterize_responsibilities_kernel<<<blocks, threads, shared_mem, stream>>>(
        C,
        n_isects,
        reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
        reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
        opacities.data_ptr<float>(),
        responsibilities.data_ptr<float>(),
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        tile_offsets.data_ptr<int32_t>(),
        flatten_ids.data_ptr<int32_t>(),
        gaussian_mass.data_ptr<float>()
    );
    return gaussian_mass;
}

} // namespace gsplat
