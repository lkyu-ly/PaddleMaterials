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

import multiprocessing
import os
import pickle
from functools import partial
from itertools import islice
from typing import List, Optional

import ase
import ase.io
from . import _keys as KEY
import numpy as np
import paddle
import tqdm
from ase.io.utils import string2index
from ase.io.vasp_parsers.vasp_outcar_parsers import (Cell,
                                                     DefaultParsersContainer,
                                                     Energy, OutcarChunkParser,
                                                     PositionsAndForces,
                                                     Stress, outcarchunks)
from ase.neighborlist import primitive_neighbor_list
from braceexpand import braceexpand
from .atom_graph_data import AtomGraphData
from pymatgen.core.structure import IStructure
from pymatgen.io.ase import AseAtomsAdaptor


def unlabeled_atoms_to_graph(atoms: ase.Atoms, cutoff: float):
    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    edge_src, edge_dst, edge_vec, shifts = primitive_neighbor_list(
        "ijDS", atoms.get_pbc(), cell, pos, cutoff, self_interaction=True
    )
    is_zero_idx = np.all(edge_vec == 0, axis=1)
    is_self_idx = edge_src == edge_dst
    non_trivials = ~(is_zero_idx & is_self_idx)
    cell_shift = np.array(shifts[non_trivials])
    edge_vec = edge_vec[non_trivials]
    edge_src = edge_src[non_trivials]
    edge_dst = edge_dst[non_trivials]
    edge_idx = np.array([edge_src, edge_dst])
    atomic_numbers = atoms.get_atomic_numbers()
    cell = np.array(cell)
    data = {
        KEY.NODE_FEATURE: atomic_numbers,
        KEY.ATOMIC_NUMBERS: atomic_numbers,
        KEY.POS: pos,
        KEY.EDGE_IDX: edge_idx,
        KEY.EDGE_VEC: edge_vec,
        KEY.CELL: cell,
        KEY.CELL_SHIFT: cell_shift,
        KEY.CELL_VOLUME: np.einsum("i,i", cell[0, :], np.cross(cell[1, :], cell[2, :])),
        KEY.NUM_ATOMS: len(atomic_numbers),
    }
    data[KEY.INFO] = {}
    return data


def atoms_to_graph(atoms: ase.Atoms, cutoff: float, transfer_info: bool = True):
    """
    From ase atoms, return AtomGraphData as graph based on cutoff radius
    Args:
        atoms (Atoms): ase atoms
        cutoff (float): cutoff radius
        transfer_info (bool): if True, transfer ".info" from atoms to graph
    Returns:
        numpy dict that can be used to initialize AtomGraphData
        by AtomGraphData(**atoms_to_graph(atoms, cutoff))
    Raises:
        RuntimeError: if ase atoms are somewhat imperfect

    Use free_energy by default (atoms.get_potential_energy(force_consistent=True))
    If it is not available, use energy (atoms.get_potential_energy())
    If stress is available, initialize stress tensor
    Ignore constraints like selective dynamics

    Requires grad is handled by 'dataset' not here.
    """
    try:
        y_energy = atoms.get_potential_energy(force_consistent=True)
    except NotImplementedError:
        y_energy = atoms.get_potential_energy()
    y_force = atoms.get_forces(apply_constraint=False)
    try:
        y_stress = -1 * atoms.get_stress()
        y_stress = np.array([y_stress[[0, 1, 2, 5, 3, 4]]])
    except RuntimeError:
        y_stress = None
    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    edge_src, edge_dst, edge_vec, shifts = primitive_neighbor_list(
        "ijDS", atoms.get_pbc(), cell, pos, cutoff, self_interaction=True
    )
    is_zero_idx = np.all(edge_vec == 0, axis=1)
    is_self_idx = edge_src == edge_dst
    non_trivials = ~(is_zero_idx & is_self_idx)
    cell_shift = np.array(shifts[non_trivials])
    edge_vec = edge_vec[non_trivials]
    edge_src = edge_src[non_trivials]
    edge_dst = edge_dst[non_trivials]
    edge_idx = np.array([edge_src, edge_dst])
    atomic_numbers = atoms.get_atomic_numbers()
    cell = np.array(cell)
    data = {
        KEY.NODE_FEATURE: atomic_numbers,
        KEY.ATOMIC_NUMBERS: atomic_numbers,
        KEY.POS: pos,
        KEY.EDGE_IDX: edge_idx,
        KEY.EDGE_VEC: edge_vec,
        KEY.ENERGY: y_energy,
        KEY.FORCE: y_force,
        KEY.STRESS: y_stress,
        KEY.CELL: cell,
        KEY.CELL_SHIFT: cell_shift,
        KEY.CELL_VOLUME: np.einsum("i,i", cell[0, :], np.cross(cell[1, :], cell[2, :])),
        KEY.NUM_ATOMS: len(atomic_numbers),
        KEY.PER_ATOM_ENERGY: y_energy / len(pos),
    }
    if transfer_info and atoms.info is not None:
        data[KEY.INFO] = atoms.info
    else:
        data[KEY.INFO] = {}
    return data


def graph_build(
    atoms_list: List,
    cutoff: float,
    num_cores: int = 1,
    transfer_info: Optional[bool] = True,
) -> List[AtomGraphData]:
    """
    parallel version of graph_build
    build graph from atoms_list and return list of AtomGraphData
    Args:
        atoms_list (List): list of ASE atoms
        cutoff (float): cutoff radius of graph
        num_cores (int, Optional): number of cores to use
        transfer_info (bool, Optional): if True, copy info from atoms to graph
    Returns:
        List[AtomGraphData]: list of AtomGraphData
    """
    serial = num_cores == 1
    inputs = [(atoms, cutoff, transfer_info) for atoms in atoms_list]
    if not serial:
        pool = multiprocessing.Pool(num_cores)
        graph_list = pool.starmap(
            atoms_to_graph, tqdm.tqdm(inputs, total=len(atoms_list))
        )
        pool.close()
        pool.join()
    else:
        graph_list = [atoms_to_graph(*input_) for input_ in inputs]
    graph_list = [AtomGraphData.from_numpy_dict(g) for g in graph_list]
    return graph_list


def ase_reader(fname, **kwargs):
    index = kwargs.pop("index", None)
    if index is None:
        index = ":"
    return ase.io.read(fname, index=index, **kwargs)


def pkl_atoms_reader(fname):
    """
    Assume the content is plane list of ase.Atoms
    """
    with open(fname, "rb") as f:
        atoms_list = []
        atoms = pickle.load(f)
        for datapoint in atoms[:100]:
            atoms = AseAtomsAdaptor.get_atoms(
                IStructure.from_dict(datapoint["structure"])
            )
            atoms_list.append(atoms)
    if type(atoms_list) != list:
        raise TypeError("The content of the pkl is not list")
    if type(atoms_list[0]) != ase.Atoms:
        raise TypeError("The content of the pkl is not list of ase.Atoms")
    return atoms_list


def structure_list_reader(filename: str, format_outputs="vasp-out"):
    parsers = DefaultParsersContainer(
        PositionsAndForces, Stress, Energy, Cell
    ).make_parsers()
    ocp = OutcarChunkParser(parsers=parsers)
    """
    Read from structure_list using braceexpand and ASE

    Args:
        fname : filename of structure_list

    Returns:
        dictionary of lists of ASE structures.
        key is title of training data (user-define)
    """

    def parse_label(line):
        line = line.strip()
        if line.startswith("[") is False:
            return False
        elif line.endswith("]") is False:
            raise ValueError("wrong structure_list title format")
        return line[1:-1]

    def parse_fileline(line):
        line = line.strip().split()
        if len(line) == 1:
            line.append(":")
        elif len(line) != 2:
            raise ValueError("wrong structure_list format")
        return line[0], line[1]

    structure_list_file = open(filename, "r")
    lines = structure_list_file.readlines()
    raw_str_dict = {}
    label = "Default"
    for line in lines:
        if line.strip() == "":
            continue
        tmp_label = parse_label(line)
        if tmp_label:
            label = tmp_label
            raw_str_dict[label] = []
            continue
        elif label in raw_str_dict:
            files_expr, index_expr = parse_fileline(line)
            raw_str_dict[label].append((files_expr, index_expr))
        else:
            raise ValueError("wrong structure_list format")
    structure_list_file.close()
    structures_dict = {}
    info_dct = {"data_from": "user_OUTCAR"}
    for title, file_lines in raw_str_dict.items():
        stct_lists = []
        for file_line in file_lines:
            files_expr, index_expr = file_line
            index = string2index(index_expr)
            for expanded_filename in list(braceexpand(files_expr)):
                f_stream = open(expanded_filename, "r")
                gen_all = outcarchunks(f_stream, ocp)
                try:
                    it_atoms = islice(gen_all, index.start, index.stop, index.step)
                except ValueError:
                    raise ValueError("Negative index is not supported yet")
                info_dct_f = {**info_dct, "file": os.path.abspath(expanded_filename)}
                for idx, o in enumerate(it_atoms):
                    try:
                        istep = index.start + idx * index.step
                        atoms = o.build()
                        atoms.info = {**info_dct_f, "ionic_step": istep}
                    except TypeError:
                        atoms = o.build()
                        atoms.info = info_dct_f
                    stct_lists.append(atoms)
                f_stream.close()
        structures_dict[title] = stct_lists
    return structures_dict


def match_reader(reader_name: str, **kwargs):
    reader = None
    metadata = {}
    if reader_name == "pkl" or reader_name == "pickle":
        reader = partial(pkl_atoms_reader, **kwargs)
        metadata.update({"origin": "atoms_pkl"})
    elif reader_name == "structure_list":
        reader = partial(structure_list_reader, **kwargs)
        metadata.update({"origin": "structure_list"})
    else:
        reader = partial(ase_reader, **kwargs)
        metadata.update({"origin": f"ase_reader"})
    return reader, metadata
