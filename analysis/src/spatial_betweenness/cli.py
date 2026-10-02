"""Command-line interface for one LAMMPS network."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import igraph as ig

from .analysis import SpatialBetweennessAnalysis
from .models import AnalysisConfig, Direction

logger = logging.getLogger(__name__)


class ProgressLogger:
    """Throttle igraph's per-source progress messages."""

    def __init__(self) -> None:
        self._message = ""
        self._percent = -10.0

    def __call__(self, message: str, percent: float) -> None:
        if message != self._message or percent >= self._percent + 5 or percent == 100:
            logger.info("%s%.0f%%", message, percent)
            self._message, self._percent = message, percent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native GEBC, OBC and DBC for an orthorhombic LAMMPS full-style data file.")
    parser.add_argument("input", type=Path, help="LAMMPS data file (Atoms # full)")
    parser.add_argument("--direction", type=Direction.parse, default=Direction.X, metavar="{x,y,z,0,1,2}")
    parser.add_argument("--dbc-length", type=float, metavar="L", help="Optional length divisor for postprocessed DBC only; raw DBC always uses distance alone")
    parser.add_argument("--output", type=Path, help="NPZ output; default: <input>.<axis>.spatial.npz")
    parser.add_argument("--validate-stock", action="store_true", help="Run a separate stock GEBC pass and require agreement")
    parser.add_argument("--inspect", action="store_true", help="Validate input and print topology without computing centrality")
    parser.add_argument("--sample-vertices", type=int, metavar="N", help="Debug only: induced neighborhood, not full-network centrality")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing NPZ and JSON output files")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = AnalysisConfig(direction=args.direction, validate_stock=args.validate_stock, dbc_length=args.dbc_length)
        analysis = SpatialBetweennessAnalysis.from_lammps(args.input, config)
        if args.sample_vertices is not None:
            analysis = analysis.sample(args.sample_vertices)
            logger.warning("Analyzing an induced debug subgraph; results are not full-network centralities")
        if args.inspect:
            print(json.dumps(analysis.inspect(), indent=2, allow_nan=False))
            return 0
        suffix = f".{config.direction.name.lower()}.spatial.npz"
        if args.sample_vertices is not None:
            suffix = f".sample{args.sample_vertices}" + suffix
        output = (args.output or args.input.with_suffix(suffix)).expanduser().resolve()
        if output.suffix != ".npz":
            raise ValueError("Output filename must end in .npz")
        if analysis.network.source in (output, output.with_suffix(".json")):
            raise ValueError("Output files must not replace the input LAMMPS file")
        if not args.overwrite and (output.exists() or output.with_suffix(".json").exists()):
            raise FileExistsError(f"Output already exists; use --overwrite or --output: {output}")
        # Only the standalone CLI owns the process-wide progress callback.
        if not args.quiet:
            ig.set_progress_handler(ProgressLogger())
        try:
            result = analysis.run()
        finally:
            if not args.quiet:
                ig.set_progress_handler(None)
        archive, report = result.save(output, overwrite=args.overwrite)
        print(json.dumps({**result.summary(), "output": str(archive), "report": str(report)}, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, ArithmeticError, ig.InternalError) as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Analysis interrupted")
        return 130
    return 0
