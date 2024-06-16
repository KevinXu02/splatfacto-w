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


class BGField(Field):
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

    def get_background_rgb(
        self, ray_bundle: RayBundle, appearance_embedding=None
    ) -> Tensor:
        """Predicts background colors at infinity."""
        directions = get_normalized_directions(ray_bundle.directions)
        directions_flat = self.direction_encoding(directions.view(-1, 3))
        if appearance_embedding is not None:
            x = torch.cat(
                [
                    directions_flat,
                    appearance_embedding.repeat(directions_flat.shape[0], 1),
                ],
                dim=-1,
            )
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
        use_view_dir=True,
        sh_levels=4,
    ):
        super().__init__()
        if use_view_dir:
            self.direction_encoding = SHEncoding(
                levels=sh_levels,
                implementation=implementation,
            )
            dir_dim=self.direction_encoding.get_out_dim()
        else:
            self.direction_encoding = None
            dir_dim=0

        # self.color_nn = MLP(
        #     in_dim=appearance_embed_dim + appearance_features_dim+dir_dim,
        #     num_layers=2,
        #     layer_width=256,
        #     out_dim=3,
        #     activation=nn.ReLU(),
        #     out_activation=nn.Sigmoid(),
        #     implementation=implementation,
        # )

        self.color_nn = nn.Sequential(
            nn.Linear(appearance_embed_dim + appearance_features_dim + dir_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        #zero init the last layer
        self.color_nn[-1].weight.data.zero_()
        self.color_nn[-1].bias.data.zero_()


    def forward(self, appearance_embed: Tensor, appearance_features: Tensor,base_color: Tensor, view_dir: Tensor = None) -> Tensor:
        if self.direction_encoding is not None:
            view_dir_flat = self.direction_encoding(view_dir.view(-1, 3))
            color_out = self.color_nn(
                torch.cat((appearance_embed, appearance_features, view_dir_flat), dim=-1)
            )
            color_out=torch.sigmoid(color_out+base_color)
        else:
            color_out = self.color_nn(
                torch.cat((appearance_embed, appearance_features), dim=-1)
            ).float()
            color_out=torch.sigmoid(color_out+base_color)
        return color_out
