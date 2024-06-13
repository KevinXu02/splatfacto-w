"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal, Optional

from torch import Tensor

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field


import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components import MLP
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.fields.base_field import Field, get_normalized_directions


class BG_Field(Field):
    def __init__(self, appearance_embedding_dim: int, implementation="torch"):
        super().__init__()
        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.mlp_background_color = MLP(
            in_dim=self.direction_encoding.get_out_dim() + appearance_embedding_dim,
            num_layers=2,
            layer_width=64,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def get_background_rgb(self, ray_bundle: RayBundle, appearance_embedding=None) -> Tensor:
        """Predicts background colors at infinity."""
        directions = get_normalized_directions(ray_bundle.directions)
        directions_flat = self.direction_encoding(directions.view(-1, 3))
        if appearance_embedding is not None:
            x = torch.cat([directions_flat, appearance_embedding.repeat(directions_flat.shape[0], 1)], dim=-1)
        else:
            x = directions_flat

        background_rgb = self.mlp_background_color(x).to(directions)

        return background_rgb

class SplatfactoWField(Field):
    def __init__(
        self,
        appearance_embed_dim,
        appearance_features_dim,
        implementation: Literal["tcnn", "torch"] = "torch",
    ):
        super().__init__()
        self.color_nn = MLP(
            in_dim=appearance_embed_dim + appearance_features_dim,
            num_layers=2,
            layer_width=256,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def forward(self, appearance_embed: Tensor, appearance_features: Tensor):
        color_out = self.color_nn(torch.cat((appearance_embed, appearance_features), dim=-1))
        return color_out
