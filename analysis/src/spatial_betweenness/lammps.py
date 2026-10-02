"""Streaming LAMMPS full-style data reader, without per-edge Python objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import PolymerNetwork


class LammpsFormatError(ValueError):
    """Invalid or unsupported LAMMPS data, with file/line context where possible."""


class LammpsDataReader:
    """Read ``Atoms # full`` and ``Bonds`` from a 3D orthorhombic data file.

    A bare ``Atoms`` header is interpreted as full style. Image flags are ignored
    because the native calculation uses minimum-image source-target distances.
    Atom IDs may be nonconsecutive. Unrelated numeric sections are skipped.
    """

    def read(self, path: str | Path) -> PolymerNetwork:
        source = Path(path).expanduser().resolve()
        counts: dict[str, int] = {}
        bounds: dict[str, tuple[float, float]] = {}
        atom_ids = coordinates = bonds = None
        nread_atoms = nread_bonds = 0
        section = None
        in_header = True
        with source.open(encoding="utf-8") as stream:
            next(stream, None)  # LAMMPS reserves the first line for the title.
            for line_number, line in enumerate(stream, start=2):
                data, _, comment = line.partition("#")
                fields = data.split()
                if not fields:
                    continue
                try:
                    if fields[-3:] == ["xy", "xz", "yz"] or fields[-1] in ("avec", "bvec", "cvec"):
                        raise ValueError("Triclinic cells are unsupported; an orthorhombic box is required")
                    if in_header and len(fields) == 2 and fields[1] in ("atoms", "bonds"):
                        key, count = fields[1], int(fields[0])
                        if count < 0 or key in counts:
                            raise ValueError(f"Invalid or repeated {key} count")
                        counts[key] = count
                        continue
                    if in_header and len(fields) == 4 and fields[2:] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
                        axis = fields[2][0]
                        if axis in bounds:
                            raise ValueError(f"Repeated {axis} box bounds")
                        bounds[axis] = (float(fields[0]), float(fields[1]))
                        continue
                    if fields[0][0].isalpha():
                        in_header = False
                        section = fields[0]
                        if section == "Atoms":
                            if "atoms" not in counts or atom_ids is not None:
                                raise ValueError("Missing atom count or repeated Atoms section")
                            style = comment.split()[0] if comment.split() else "full"
                            if style != "full":
                                raise ValueError(f"Unsupported atom style {style!r}; expected Atoms # full")
                            atom_ids = np.empty(counts["atoms"], dtype=np.int64)
                            coordinates = np.empty((counts["atoms"], 3), dtype=np.float64)
                        elif section == "Bonds":
                            if bonds is not None:
                                raise ValueError("Repeated Bonds section")
                            bonds = np.empty((counts.get("bonds", 0), 4), dtype=np.int64)
                        continue
                    if section == "Atoms":
                        if len(fields) not in (7, 10) or nread_atoms >= len(atom_ids):
                            raise ValueError("Expected exactly the declared number of full-style atom rows (7 or 10 columns)")
                        atom_ids[nread_atoms] = int(fields[0])
                        coordinates[nread_atoms] = [float(value) for value in fields[4:7]]
                        nread_atoms += 1
                    elif section == "Bonds":
                        if len(fields) != 4 or nread_bonds >= len(bonds):
                            raise ValueError("Expected exactly the declared number of bond rows (4 columns)")
                        bonds[nread_bonds] = [int(value) for value in fields]
                        nread_bonds += 1
                except (ValueError, OverflowError) as exc:
                    raise LammpsFormatError(f"{source}:{line_number}: {exc}") from exc

        if "atoms" not in counts or nread_atoms != counts["atoms"] or nread_bonds != counts.get("bonds", 0):
            raise LammpsFormatError(f"{source}: Atoms/Bonds sections do not match the declared counts")
        if set(bounds) != {"x", "y", "z"}:
            raise LammpsFormatError(f"{source}: missing orthorhombic box bounds")
        if atom_ids is None:
            atom_ids = np.empty(0, dtype=np.int64)
            coordinates = np.empty((0, 3), dtype=np.float64)
        if bonds is None:
            bonds = np.empty((0, 4), dtype=np.int64)
        order = np.argsort(atom_ids)
        atom_ids, coordinates = atom_ids[order], coordinates[order]
        if np.any(atom_ids <= 0) or np.any(np.diff(atom_ids) <= 0):
            raise LammpsFormatError(f"{source}: atom IDs must be positive and unique")
        edges = np.searchsorted(atom_ids, bonds[:, 2:4])
        if edges.size and (np.any(edges >= atom_ids.size) or not np.array_equal(atom_ids[edges], bonds[:, 2:4])):
            raise LammpsFormatError(f"{source}: a bond references an unknown atom ID")
        try:
            return PolymerNetwork(atom_ids, coordinates, bonds[:, 0], bonds[:, 1], edges,
                                  np.array([bounds[axis] for axis in "xyz"]), source)
        except ValueError as exc:
            raise LammpsFormatError(f"{source}: {exc}") from exc
