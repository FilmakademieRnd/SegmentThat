# -------------------------------------------------------------------------------
# SegmentThat: Interactive Gaussian Splat Segmentation
#
# Copyright (C) 2026 Filmakademie Baden-Württemberg,
# Carl Laemmle Institut for Research and Media Innovation
# https://github.com/FilmakademieRnd/SegmentThat_InteractiveTool
#
# SegmentThat is developed by the Carl Laemmle Institut for Research and Media
# Innovation in the context of the Industrial Metaverse Reallabor project
# (https://cli.filmakademie.de/imrl), funded by the Ministry of Economic Affairs,
# Skilled Crafts and Tourism Baden-Württemberg.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the MIT License for more details.
# You should have received a copy of the MIT License along with this program;
# if not go to https://opensource.org/licenses/MIT
# -------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
#
# Author: Sevinc Eroglu
# Date: 27.10.2026

"""Minimal reader and writer for binary 3D Gaussian Splatting .ply files.

Only the `vertex` element is kept. ASCII files and list properties are not supported,
since 3DGS files do not use them.
"""

# Standard library
from dataclasses import dataclass, field

# Third-party
import numpy as np

PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
PLY_TYPE_NAMES = {
    "i1": "char",
    "u1": "uchar",
    "i2": "short",
    "u2": "ushort",
    "i4": "int",
    "u4": "uint",
    "f4": "float",
    "f8": "double",
}
BYTE_ORDERS = {"binary_little_endian": "<", "binary_big_endian": ">"}


@dataclass
class Ply:
    vertices: np.ndarray
    byte_order: str = "<"
    # Header comments before the first element, and those inside the vertex element.
    comments: list = field(default_factory=list)
    vertex_comments: list = field(default_factory=list)
    obj_info: list = field(default_factory=list)


def read_ply(path):
    """Read the vertex element of a binary .ply file as a numpy structured array."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        byte_order = None
        comments, obj_info, elements = [], [], []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            line = line.decode("ascii").strip()
            if line == "end_header":
                break
            keyword, _, rest = line.partition(" ")
            if keyword == "format":
                fmt = rest.split()[0]
                if fmt not in BYTE_ORDERS:
                    raise ValueError(f"Unsupported PLY format '{fmt}' (only binary is supported).")
                byte_order = BYTE_ORDERS[fmt]
            elif keyword == "comment":
                (elements[-1][3] if elements else comments).append(rest)
            elif keyword == "obj_info":
                obj_info.append(rest)
            elif keyword == "element":
                name, count = rest.split()
                elements.append((name, int(count), [], []))
            elif keyword == "property":
                parts = rest.split()
                if parts[0] == "list":
                    raise ValueError("PLY list properties are not supported.")
                if parts[0] not in PLY_TYPES:
                    raise ValueError(f"Unknown PLY property type '{parts[0]}'.")
                elements[-1][2].append((parts[1], byte_order + PLY_TYPES[parts[0]]))
        if byte_order is None:
            raise ValueError(f"PLY header has no format line: {path}")
        for name, count, props, element_comments in elements:
            dtype = np.dtype(props)
            if name != "vertex":
                f.seek(count * dtype.itemsize, 1)
                continue
            vertices = np.fromfile(f, dtype=dtype, count=count)
            if len(vertices) != count:
                raise ValueError(f"PLY file is truncated: {path}")
            return Ply(vertices, byte_order, comments, element_comments, obj_info)
    raise ValueError(f"PLY file has no vertex element: {path}")


def write_ply(path, vertices, byte_order="<", comments=(), vertex_comments=(), obj_info=()):
    """Write a numpy structured array as the vertex element of a binary .ply file."""
    fmt = {v: k for k, v in BYTE_ORDERS.items()}[byte_order]
    lines = ["ply", f"format {fmt} 1.0"]
    lines += [f"comment {c}" for c in comments]
    lines += [f"obj_info {o}" for o in obj_info]
    lines.append(f"element vertex {len(vertices)}")
    lines += [f"comment {c}" for c in vertex_comments]
    props = []
    for name in vertices.dtype.names:
        kind = vertices.dtype[name].str[1:]
        lines.append(f"property {PLY_TYPE_NAMES[kind]} {name}")
        props.append((name, byte_order + kind))
    lines.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        vertices.astype(np.dtype(props), copy=False).tofile(f)
