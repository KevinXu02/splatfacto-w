"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components import MLP
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.fields.base_field import Field, get_normalized_directions
from gsplat.cuda._wrapper import spherical_harmonics


class BGField(Field):
    def __init__(
        self, appearance_embedding_dim: int, implementation="torch", sh_levels=4
    ):
        super().__init__()
        self.sh_dim = (sh_levels + 1) ** 2

        self.encoder = nn.Sequential(
            nn.Linear(appearance_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.sh_base_head = nn.Linear(128, 3)
        self.sh_rest_head = nn.Linear(128, (self.sh_dim - 1) * 3)
        # zero initialization
        self.sh_rest_head.weight.data.zero_()
        self.sh_rest_head.bias.data.zero_()

    def get_background_rgb(
        self, ray_bundle: RayBundle, appearance_embedding=None, num_sh=4
    ) -> Tensor:
        """Predicts background colors at infinity."""
        cur_sh_dim = (num_sh + 1) ** 2
        directions = ray_bundle.directions.view(-1, 3)
        x = self.encoder(appearance_embedding)
        sh_base = self.sh_base_head(x)  # [batch, 3]
        sh_rest = self.sh_rest_head(x)[
            ..., : (cur_sh_dim - 1) * 3
        ]  # [batch, 3 * (num_sh - 1)]
        sh_coeffs = (
            torch.cat([sh_base, sh_rest], dim=-1)
            .view(-1, cur_sh_dim, 3)
            .repeat(directions.shape[0], 1, 1)
        )
        colors = spherical_harmonics(
            degrees_to_use=num_sh, dirs=directions, coeffs=sh_coeffs
        )

        return colors


class SplatfactoWField(Field):
    def __init__(
        self,
        appearance_embed_dim,
        appearance_features_dim,
        implementation: Literal["tcnn", "torch"] = "torch",
        sh_levels=4,
    ):
        super().__init__()

        # self.color_nn = MLP(
        #     in_dim=appearance_embed_dim + appearance_features_dim+dir_dim,
        #     num_layers=2,
        #     layer_width=256,
        #     out_dim=3,
        #     activation=nn.ReLU(),
        #     out_activation=nn.Sigmoid(),
        #     implementation=implementation,
        # )
        self.sh_dim = (sh_levels + 1) ** 2
        self.encoder = nn.Sequential(
            nn.Linear(appearance_embed_dim + appearance_features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.sh_base_head = nn.Linear(256, 3)
        self.sh_rest_head = nn.Linear(256, (self.sh_dim - 1) * 3)
        # zero initialization
        self.sh_rest_head.weight.data.zero_()
        self.sh_rest_head.bias.data.zero_()

    def forward(
        self,
        appearance_embed: Tensor,
        appearance_features: Tensor,
    ) -> Tensor:
        x = self.encoder(torch.cat((appearance_embed, appearance_features), dim=-1))
        base_color = self.sh_base_head(x)
        sh_rest = self.sh_rest_head(x)
        sh_coeffs = torch.cat([base_color, sh_rest], dim=-1).view(-1, self.sh_dim, 3)
        return sh_coeffs
