#!/usr/bin/env bash

# Batch training-and-render script for SeaFree-GS.
# - scene_list: relative scene paths under the repository-level data/ directory.
# - EXP_TAG: experiment label used to group outputs/ and render_results/ entries.
# - The script assumes the default monocular depth folder is depthAnything_u16.

set -u -o pipefail

clear

# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=1

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT_DIR="${PROJECT_ROOT}/outputs"
RENDER_ROOT="${PROJECT_ROOT}/render_results"

#
# Example scenes for the public release.
# Replace these with your own relative scene paths under the repository-level data/ directory as needed.
#
scene_list=(
    "SeaThru/D3"
    "SeaThru/D5"
    "Seathru_NeRF_Undistortion/Curasao"
    "Seathru_NeRF_Undistortion/IUI3_RedSea"
    "Seathru_NeRF_Undistortion/JapaneseGradens_RedSea"
    "Seathru_NeRF_Undistortion/Panama"
)

EXP_TAG="seafree_gs_release"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

find_latest_config() {
    local run_root="$1"
    local latest_dir=""

    if [ ! -d "$run_root" ]; then
        return 1
    fi

    latest_dir=$(ls -td "$run_root"/*/ 2>/dev/null | head -n 1 || true)
    if [ -z "$latest_dir" ]; then
        return 1
    fi

    echo "${latest_dir%/}/config.yml"
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

if [ "${#scene_list[@]}" -eq 0 ]; then
    echo "[WARN] scene_list is empty. Uncomment or add scene paths before running this script."
    exit 0
fi

echo "######################################################################################################################################################"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Log root: ${LOG_ROOT_DIR}"
echo "[INFO] Render root: ${RENDER_ROOT}"
echo "######################################################################################################################################################"

for scene_rel_path in "${scene_list[@]}"; do
    data_name="${scene_rel_path%/*}"
    scene_name="${scene_rel_path##*/}"
    DATA_DIR="${PROJECT_ROOT}/data/${data_name}/${scene_name}"
    EXP_NAME="${EXP_TAG}/${data_name}/${scene_name}"
    LOG_DIR="${LOG_ROOT_DIR}/${EXP_NAME}/seafree-gs"
    RENDER_OUTPUT_DIR="${RENDER_ROOT}/${EXP_TAG}/${data_name}/${scene_name}/seafree-gs"

    echo "######################################################################################################################################################"
    echo "[INFO] Scene: ${scene_rel_path}"
    echo "[INFO] Data directory: ${DATA_DIR}"
    echo "[INFO] Experiment name: ${EXP_NAME}"
    echo "[INFO] Render output directory: ${RENDER_OUTPUT_DIR}"

    mkdir -p "${RENDER_OUTPUT_DIR}"

    if ! ns-train seafree-gs --experiment-name "${EXP_NAME}" --vis tensorboard \
    --data "${DATA_DIR}" \
    colmap \
    --colmap-path sparse/0 \
    --images-path images_wb \
    --depths-path depthAnything_u16; then
        echo "[ERROR] Training failed for scene: ${scene_rel_path}"
        continue
    fi

    latest_config="$(find_latest_config "${LOG_DIR}")" || true

    if [ -z "${latest_config}" ]; then
        echo "[ERROR] Failed to locate the latest config.yml under: ${LOG_DIR}"
        continue
    fi

    echo "[INFO] Latest config: ${latest_config}"

    # Dataset render examples on the default evaluation split.
    ns-render dataset --load-config "${latest_config}" --rendered-output-names rgb --output-path "${RENDER_OUTPUT_DIR}"
    ns-render dataset --load-config "${latest_config}" --rendered-output-names intrinsic_color_render --output-path "${RENDER_OUTPUT_DIR}"
    ns-render dataset --load-config "${latest_config}" --rendered-output-names depth --output-path "${RENDER_OUTPUT_DIR}" --colormap-options.colormap inferno

    # Dataset render examples on the training split.
    ns-render dataset --load-config "${latest_config}" --split train --rendered-output-names rgb --output-path "${RENDER_OUTPUT_DIR}"
    ns-render dataset --load-config "${latest_config}" --split train --rendered-output-names intrinsic_color_render --output-path "${RENDER_OUTPUT_DIR}"
    ns-render dataset --load-config "${latest_config}" --split train --rendered-output-names depth --output-path "${RENDER_OUTPUT_DIR}" --colormap-options.colormap inferno

    # Interpolation render of the reconstructed true-appearance view.
    ns-render interpolate --load-config "${latest_config}" --interpolation-steps 80 --rendered-output-names intrinsic_color_render --output-path "${RENDER_OUTPUT_DIR}/intrinsic_color_render.mp4"
done

echo "######################################################################################################################################################"
echo "[INFO] Batch run finished."
