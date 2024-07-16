"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from splatfactow.splatfactow_datamanager import (
    SplatfactoWDatamanagerConfig,
)
from splatfactow.splatfactow_model import SplatfactoWModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from splatfactow.nerfw_dataparser import NerfWDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


splatfactow_config = MethodSpecification(
    description="Splatfacto in the wild",
    config=TrainerConfig(
        method_name="splatfacto-w",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=55000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=SplatfactoWDatamanagerConfig(
                dataparser=NerfWDataParserConfig(),
                cache_images_type="uint8",
            ),
            model=SplatfactoWModelConfig(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-7,
                    max_steps=30000,
                ),
            },
            "appearance_features": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=40000,
                ),
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.03, eps=1e-15),
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
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            },
            "field_background_encoder": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000
                ),
            },
            "field_background_base": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-4, max_steps=30000
                ),
            },
            "field_background_rest": {
                "optimizer": AdamOptimizerConfig(lr=2e-3 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-4 / 20, max_steps=30000
                ),
            },
            "appearance_model_encoder": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000
                ),
            },
            "appearance_model_base": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000
                ),
            },
            "appearance_model_rest": {
                "optimizer": AdamOptimizerConfig(lr=2e-3 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4 / 20, max_steps=30000
                ),
            },
            "appearance_embed": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=3e-4, max_steps=40000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
)

splatfactow_light_config = MethodSpecification(
    description="Splatfacto in the wild (light)",
    config=TrainerConfig(
        method_name="splatfacto-w-light",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=10000000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    load_3D_points=True,
                ),
                cache_images_type="uint8",
            ),
            model=SplatfactoWModelConfig(
                appearance_embed_dim=24,
                appearance_features_dim=32,
                app_layer_width=128,
                app_num_layers=2,
                bg_layer_width=128,
                bg_num_layers=2,
                sh_degree_interval=1000,
                bg_sh_degree=4,
                enable_bg_model=False,
                enable_alpha_loss=False,
                enable_robust_mask=False,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-7,
                    max_steps=30000,
                ),
            },
            "appearance_features": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=40000,
                ),
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.03, eps=1e-15),
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
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=15000
                ),
            },
            "field_background_encoder": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=15000
                ),
            },
            "field_background_base": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-4, max_steps=15000
                ),
            },
            "field_background_rest": {
                "optimizer": AdamOptimizerConfig(lr=2e-3 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-4 / 20, max_steps=15000
                ),
            },
            "appearance_model_encoder": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=15000
                ),
            },
            "appearance_model_base": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=15000
                ),
            },
            "appearance_model_rest": {
                "optimizer": AdamOptimizerConfig(lr=2e-3 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4 / 20, max_steps=15000
                ),
            },
            "appearance_embed": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=3e-4, max_steps=15000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
)
