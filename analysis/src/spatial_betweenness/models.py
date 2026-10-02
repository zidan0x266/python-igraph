"""Typed configuration and atom/bond data for spatial centrality analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from numbers import Real
from operator import index
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class Direction(IntEnum):
    """Loading axis; values match the native igraph API."""

    X = 0
    Y = 1
    Z = 2

    @classmethod
    def parse(cls, value: Direction | str | int) -> Direction:
        if isinstance(value, str):
            value = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}.get(value.strip().lower(), -1)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Direction must be x, y, z, 0, 1 or 2")
        try:
            return cls(index(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("Direction must be x, y, z, 0, 1 or 2") from exc


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Native analysis settings; dbc_length is used only for output postprocessing.

    None leaves DBC without any reference-length division. Raw DBC always uses
    source-target distance alone, regardless of this setting.
    """

    direction: Direction | str | int = Direction.X
    validate_stock: bool = False
    rtol: float = 1e-12
    atol: float = 1e-8
    dbc_length: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", Direction.parse(self.direction))
        if self.dbc_length is not None:
            if isinstance(self.dbc_length, (bool, np.bool_)) or not isinstance(self.dbc_length, Real):
                raise ValueError("dbc_length must be a finite, positive number or None")
            try:
                length = float(self.dbc_length)
            except OverflowError as exc:
                raise ValueError("dbc_length must be finite and positive") from exc
            if not np.isfinite(length) or length <= 0:
                raise ValueError("dbc_length must be finite and positive")
            object.__setattr__(self, "dbc_length", length)
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _integers(values: IntArray, name: str) -> IntArray:
    array = np.asarray(values)
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    if array.size and np.issubdtype(array.dtype, np.unsignedinteger) and array.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds the int64 range")
    return np.array(array, dtype=np.int64, copy=True)


@dataclass(frozen=True, slots=True)
class PolymerNetwork:
    """An orthorhombic bead-level graph with explicit LAMMPS ID mapping.

    Coordinate rows follow sorted ``atom_ids``. ``edges`` contains zero-based
    indices into those rows. Bond arrays retain their original file order.
    Arrays are copied and made read-only so analysis cannot silently alter input.
    """

    atom_ids: IntArray
    coordinates: FloatArray
    bond_ids: IntArray
    bond_types: IntArray
    edges: IntArray
    box_bounds: FloatArray
    source: Path | None = None

    def __post_init__(self) -> None:
        for name in ("atom_ids", "bond_ids", "bond_types", "edges"):
            object.__setattr__(self, name, _integers(getattr(self, name), name))
        for name in ("coordinates", "box_bounds"):
            object.__setattr__(self, name, np.array(getattr(self, name), dtype=np.float64, copy=True))
        if self.atom_ids.ndim != 1 or self.bond_ids.ndim != 1:
            raise ValueError("Atom IDs and bond IDs must be one-dimensional")
        if self.coordinates.shape != (self.n_atoms, 3) or not np.isfinite(self.coordinates).all():
            raise ValueError("Coordinates must be a finite (n_atoms, 3) array")
        if np.any(self.atom_ids <= 0) or np.any(np.diff(self.atom_ids) <= 0):
            raise ValueError("Atom IDs must be positive, unique and sorted")
        if np.any(self.bond_ids <= 0) or np.unique(self.bond_ids).size != self.n_bonds:
            raise ValueError("Bond IDs must be positive and unique")
        if self.bond_types.shape != (self.n_bonds,) or np.any(self.bond_types <= 0):
            raise ValueError("Each bond must have a positive bond type")
        if self.edges.shape != (self.n_bonds, 2):
            raise ValueError("Edges must have shape (n_bonds, 2)")
        if self.edges.size and (self.edges.min() < 0 or self.edges.max() >= self.n_atoms):
            raise ValueError("Edge endpoint outside the vertex-index range")
        if self.box_bounds.shape != (3, 2) or not np.isfinite(self.box_bounds).all():
            raise ValueError("Box bounds must be a finite (3, 2) array")
        if not np.isfinite(self.box_lengths).all() or np.any(self.box_lengths <= 0):
            raise ValueError("Box lengths must be finite and strictly positive")
        if self.source is not None:
            object.__setattr__(self, "source", Path(self.source).resolve())
        for name in ("atom_ids", "coordinates", "bond_ids", "bond_types", "edges", "box_bounds"):
            getattr(self, name).setflags(write=False)

    @property
    def n_atoms(self) -> int:
        return self.atom_ids.size

    @property
    def n_bonds(self) -> int:
        return self.bond_ids.size

    @property
    def box_lengths(self) -> FloatArray:
        return self.box_bounds[:, 1] - self.box_bounds[:, 0]

    @property
    def bond_atom_ids(self) -> IntArray:
        """Original LAMMPS endpoint IDs, with endpoint and bond-row order preserved."""
        return self.atom_ids[self.edges]

    def subset(self, vertices: IntArray) -> PolymerNetwork:
        """Build an induced debug subgraph; its centralities differ from the full graph."""
        vertices = np.unique(_integers(vertices, "vertices"))
        if vertices.size and (vertices.min() < 0 or vertices.max() >= self.n_atoms):
            raise ValueError("Sample vertex index outside the graph")
        selected = np.zeros(self.n_atoms, dtype=bool)
        selected[vertices] = True
        keep = selected[self.edges].all(axis=1)
        edges = np.searchsorted(vertices, self.edges[keep])
        return PolymerNetwork(self.atom_ids[vertices], self.coordinates[vertices], self.bond_ids[keep],
                              self.bond_types[keep], edges, self.box_bounds, self.source)
