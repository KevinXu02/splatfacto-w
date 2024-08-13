# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

import typing
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import argparse
import numpy as np
import torch

from nerfstudio.data.scene_box import OrientedBox
from splatfactow.splatfactow_model import SplatfactoWModel
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    camera_idx: Optional[int] = None
    """Index of the camera to use for rendering. If None, the first camera is used."""

    @staticmethod
    def write_ply(
        filename: str,
        count: int,
        map_to_tensors: typing.OrderedDict[str, np.ndarray],
    ):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Note: All float values will be converted to float32 for writing.

        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        """

        # Ensure count matches the length of all tensors
        if not all(len(tensor) == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float or uint8 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError(
                "All tensors must be numpy arrays of float or uint8 type and not empty"
            )

        with open(filename, "wb") as ply_file:
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")

            ply_file.write(f"element vertex {count}\n".encode())

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            # Write binary data
            # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoWModel)

        model: SplatfactoWModel = pipeline.model

        filename = self.output_dir / "splat.ply"

        count = 0
        map_to_tensors = OrderedDict()

        with torch.no_grad():
            model.set_camera_idx(self.camera_idx)
            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            n = count
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                raise ValueError("SH degree must be greater than 0")

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

            if (
                self.obb_center is not None
                and self.obb_rotation is not None
                and self.obb_scale is not None
            ):
                crop_obb = OrientedBox.from_params(
                    self.obb_center, self.obb_rotation, self.obb_scale
                )
                assert crop_obb is not None
                mask = crop_obb.within(torch.from_numpy(positions)).numpy()
                for k, t in map_to_tensors.items():
                    map_to_tensors[k] = map_to_tensors[k][mask]

                n = map_to_tensors["x"].shape[0]
                count = n

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            CONSOLE.print(
                f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}"
            )
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)

        ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a Gaussian Splat model to a .ply"
    )
    parser.add_argument(
        "--load_config",
        type=Path,
        help="Path to the config YAML file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--obb_center",
        type=str,
        help="Center of the oriented bounding box.",
    )
    parser.add_argument(
        "--obb_rotation",
        type=str,
        help="Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians",
    )
    parser.add_argument(
        "--obb_scale",
        type=str,
        help="Scale of the oriented bounding box along each axis.",
    )
    parser.add_argument(
        "--camera_idx",
        type=int,
        help="Index of the camera to use for rendering. If None, the first camera is used.",
    )

    args = parser.parse_args()

    obb_center = (
        tuple(map(float, args.obb_center.split(","))) if args.obb_center else None
    )
    obb_rotation = (
        tuple(map(float, args.obb_rotation.split(","))) if args.obb_rotation else None
    )
    obb_scale = tuple(map(float, args.obb_scale.split(","))) if args.obb_scale else None

    exporter = ExportGaussianSplat(
        load_config=args.load_config,
        output_dir=args.output_dir,
        obb_center=obb_center,
        obb_rotation=obb_rotation,
        obb_scale=obb_scale,
        camera_idx=args.camera_idx,
    )
    exporter.main()
