"""Nerfstudio method registration for SeaFree-GS."""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from seafree_gs.seafree_datamanager import SeaFreeGsFullImageDatamanager, SeaFreeGsFullImageDatamanagerConfig
from seafree_gs.seafree_dataparser import SeaFreeGsDataParserConfig
from seafree_gs.seafree_model import SeaFreeGsModelConfig


SeaFreeGsMethod = MethodSpecification(
    config=TrainerConfig(
        method_name="seafree-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=SeaFreeGsFullImageDatamanagerConfig(
                _target=SeaFreeGsFullImageDatamanager[DepthDataset],
                dataparser=SeaFreeGsDataParserConfig(load_3D_points=True),
            ),
            model=SeaFreeGsModelConfig(
                output_depth_during_training=True,
                rasterize_mode="antialiased",
                sh_degree=0,
                reset_alpha_value=0.5,
                cull_alpha_thresh=0.5,
                cull_alpha_thresh_post=0.1,
                continue_cull_post_densification=True,
                reset_alpha_every=5,                
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7,
                    max_steps=30000,
                    warmup_steps=1000,
                    lr_pre_warmup=0,
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4,
                    max_steps=30000,
                    warmup_steps=1000,
                    lr_pre_warmup=0,
                ),
            },
            "water_properties_predictor": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4,
                    max_steps=30000,
                ),
            },
            "water_transport_field": {
                "optimizer": AdamOptimizerConfig(lr=3e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=8e-5,
                    max_steps=23000,
                    warmup_steps=15000,
                    lr_pre_warmup=0,
                ),
            },
            "line_of_sight_direction_encoding": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4,
                    max_steps=30000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="SeaFree-GS",
)
