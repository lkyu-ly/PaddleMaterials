# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Adapted from https://github.com/divelab/AIRS (OpenMat/HIENet)

"""HIENetDataset: extxyz frames -> getitem dict of {graph, labels, id}.

The getitem contract returns graph + labels + id. Labels travel the dict
level (batched by DefaultCollator
recursion), not as Data attributes: inside a ppmat Batch a 2D stress
would be flattened to (3*B, 3), and bare-ndarray force with varying N
cannot np.stack -- so force is a ConcatNumpyWarper([N,3]) (concatenate
branch) and energy a scalar np.float64 (numbers branch -> [B]).

Note: dataset-level statistics (scale/shift/conv_denominator) are NOT
computed here; they come from the checkpoint config.
"""

import paddle
from ase.io import read

from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.models import build_graph_converter


class HIENetDataset(paddle.io.Dataset):
    """Read all frames once at construction; atoms cached at init, graphs
    built per getitem. __getitem__ converts each ase Atoms into the
    contract dict.

    ``converter`` accepts the built object or its
    ``{__class_name__, __init_params__}`` cfg (the ``${Global.graph_converter}``
    yaml form).
    ``frame_offset``/``num_frames`` select a contiguous slice of the file,
    splitting one extxyz into train/val frames.
    """

    def __init__(
        self,
        path: str,
        converter,
        frame_offset: int = 0,
        num_frames: int | None = None,
        **kwargs,  # for compatibility
    ):
        super().__init__()
        self.path = path
        if isinstance(converter, dict):
            converter = build_graph_converter(converter)
        self.converter = converter
        self.frame_offset = int(frame_offset)
        atoms_list = read(path, index=":")
        if num_frames is not None:
            atoms_list = atoms_list[
                self.frame_offset : self.frame_offset + int(num_frames)
            ]
        self.atoms_list = atoms_list

    def __getitem__(self, i) -> dict:
        data = self.converter(self.atoms_list[i])
        sample = {"graph": data, "id": int(i) + self.frame_offset}
        if "energy" in data:
            # Scalar at dict level: DefaultCollator numbers branch -> [B].
            sample["energy"] = data["energy"][0]
        if "force" in data:
            # Concatenate branch -> [sum_N, 3] in the batch.
            sample["force"] = ConcatNumpyWarper(data["force"])
        if "stress" in data:
            # Fixed [3,3] ndarray: np.stack branch -> [B, 3, 3].
            sample["stress"] = data["stress"]
        return sample

    def __len__(self) -> int:
        return len(self.atoms_list)
