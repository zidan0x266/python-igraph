"""Centrality results, stock comparisons and portable NumPy output."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .models import AnalysisConfig, FloatArray, PolymerNetwork


@dataclass(frozen=True, slots=True)
class ValidationReport:
    max_absolute_difference: float
    mean_absolute_difference: float
    max_relative_difference: float
    pearson: float | None
    allclose: bool
    rtol: float
    atol: float
    relative_denominator_floor: float = 1e-15

    @classmethod
    def compare(cls, actual: FloatArray, reference: FloatArray, *, rtol: float, atol: float) -> ValidationReport:
        if actual.shape != reference.shape:
            raise ValueError("Centrality arrays must have identical shapes")
        if not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ArithmeticError("Cannot validate nonfinite centrality values")
        difference = np.abs(actual - reference)
        pearson = None
        if actual.size > 1 and np.std(actual) > 0 and np.std(reference) > 0:
            pearson = float(np.corrcoef(actual, reference)[0, 1])
        return cls(float(difference.max(initial=0)), float(difference.mean()) if difference.size else 0.0,
                   float((difference / np.maximum(np.abs(reference), 1e-15)).max(initial=0)), pearson,
                   bool(np.allclose(actual, reference, rtol=rtol, atol=atol)), rtol, atol)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CentralityResult:
    """Raw undirected GEBC/OBC/DBC arrays aligned with ``network.bond_ids``."""

    network: PolymerNetwork
    raw_gebc: FloatArray
    raw_obc: FloatArray
    raw_dbc: FloatArray
    neff: int
    metadata: dict[str, object]
    validation: ValidationReport | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.neff <= self.network.n_atoms:
            raise ValueError("neff is outside the network's vertex range")
        for name in ("raw_gebc", "raw_obc", "raw_dbc"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if values.shape != (self.network.n_bonds,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ArithmeticError(f"{name} must contain one finite, nonnegative value per bond")
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def normalization_factor(self) -> float:
        return 2.0 / (self.neff * (self.neff - 1)) if self.neff > 1 else 0.0

    @property
    def obc(self) -> FloatArray:
        """OBC normalized by the number of unordered pairs of nonisolated nodes."""
        return self.raw_obc * self.normalization_factor

    @property
    def dbc(self) -> FloatArray:
        """Pair-normalized DBC, optionally divided by the selected postprocessing length."""
        length = self.metadata.get("dbc_length")
        if length is not None:
            return self.dbc_for_length(length, normalized=True)
        return self.raw_dbc * self.normalization_factor

    def dbc_for_length(self, length: float, *, normalized: bool = False) -> FloatArray:
        """Apply any positive reference length to the same unmodified raw DBC array."""
        if length is None:
            raise ValueError("dbc_length must be finite and positive")
        length = AnalysisConfig(dbc_length=length).dbc_length
        with np.errstate(over="raise", invalid="raise"):
            values = self.raw_dbc / length
        return values * self.normalization_factor if normalized else values

    def summary(self) -> dict[str, object]:
        report = dict(self.metadata)
        report.update(neff=self.neff, normalization_factor=self.normalization_factor,
                      gebc_validation=self.validation.to_dict() if self.validation else None)
        return report

    def save(self, path: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
        """Write a pickle-free NPZ and JSON report, using temporary files before replacement.

        ``edges`` indexes ``atom_ids``; ``bond_atom_ids`` stores actual LAMMPS IDs.
        No extra factor of 1/2 is applied. Existing outputs require overwrite=True.
        """
        output = Path(path).expanduser().resolve()
        if output.suffix != ".npz":
            raise ValueError("Output filename must end in .npz")
        report_path = output.with_suffix(".json")
        if self.network.source in (output, report_path):
            raise ValueError("Output files must not replace the input LAMMPS file")
        if not overwrite and (output.exists() or report_path.exists()):
            raise FileExistsError(f"Output already exists: {output} or {report_path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(self.summary(), indent=2, allow_nan=False)
        temporary: list[Path] = []
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as stream:
                temporary.append(Path(stream.name))
                np.savez_compressed(stream, atom_ids=self.network.atom_ids, coords=self.network.coordinates,
                                    bond_ids=self.network.bond_ids, bond_types=self.network.bond_types,
                                    edges=self.network.edges, bond_atom_ids=self.network.bond_atom_ids,
                                    box_bounds=self.network.box_bounds, box_lengths=self.network.box_lengths,
                                    direction=self.metadata["direction"], neff=self.neff,
                                    dbc_length=self.metadata["dbc_length"] if self.metadata["dbc_length"] is not None else 1.0,
                                    dbc_length_applied=self.metadata["dbc_length"] is not None,
                                    raw_dbc_weight="distance", schema_version=2,
                                    raw_GEBC_igraph=self.raw_gebc, raw_OBC=self.raw_obc, raw_DBC=self.raw_dbc,
                                    OBC=self.obc, DBC=self.dbc, metadata=metadata)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=f".{report_path.name}.", delete=False) as stream:
                temporary.append(Path(stream.name))
                stream.write(metadata + "\n")
            os.replace(temporary[0], output)
            os.replace(temporary[1], report_path)
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)
        return output, report_path
