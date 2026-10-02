#!/usr/bin/env python3
"""Compare distance-only native DBC with a saved distance/L reference archive."""

import argparse
import json
from pathlib import Path

import igraph as ig
import numpy as np

from spatial_betweenness import AnalysisConfig, SpatialBetweennessAnalysis, ValidationReport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("reference", type=Path, help="Old NPZ with distance/L raw_DBC")
    parser.add_argument("--output", type=Path, required=True, help="New distance-only NPZ (must not exist)")
    parser.add_argument("--validate-stock", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    with np.load(args.reference, allow_pickle=False) as archive:
        old = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(old["metadata"]))
    if metadata.get("raw_dbc_weight") == "distance":
        raise ValueError("Reference already uses distance-only DBC; expected old distance/L outputs")
    config = AnalysisConfig(direction=int(old["direction"]), validate_stock=args.validate_stock)
    analysis = SpatialBetweennessAnalysis.from_lammps(args.input, config)
    for name, values in (("atom_ids", analysis.network.atom_ids), ("bond_ids", analysis.network.bond_ids),
                         ("edges", analysis.network.edges), ("coords", analysis.network.coordinates),
                         ("box_lengths", analysis.network.box_lengths)):
        np.testing.assert_array_equal(values, old[name], err_msg=f"Reference/input mismatch: {name}")
    length = metadata.get("dbc_length")
    if length is None:
        length = float(old["box_lengths"][int(old["direction"])])
    length = AnalysisConfig(dbc_length=length).dbc_length
    last = [None, -5.0]

    def progress(message: str, percent: float) -> None:
        if message != last[0] or percent >= last[1] + 5 or percent == 100:
            print(f"{message}{percent:.0f}%", flush=True)
            last[:] = [message, percent]

    ig.set_progress_handler(progress)
    try:
        result = analysis.run()
    finally:
        ig.set_progress_handler(None)

    def compare(actual, reference):
        return {**ValidationReport.compare(actual, reference, rtol=1e-12, atol=1e-8).to_dict(),
                "exact_array_equality": bool(np.array_equal(actual, reference))}

    checks = {
        "gebc_unchanged": compare(result.raw_gebc, old["raw_GEBC_igraph"]),
        "obc_unchanged": compare(result.raw_obc, old["raw_OBC"]),
        "new_raw_dbc_equals_old_times_old_length": compare(result.raw_dbc, old["raw_DBC"] * length),
        "old_raw_dbc_recovered": compare(result.dbc_for_length(length), old["raw_DBC"]),
        "old_normalized_dbc_recovered": compare(result.dbc_for_length(length, normalized=True), old["DBC"]),
    }
    for applied in (1.0, 2.5, 49.39694662):
        checks[f"dbc_at_length_{applied}"] = compare(result.dbc_for_length(applied), old["raw_DBC"] * (length / applied))
    report = {"input": str(args.input), "reference": str(args.reference), "reference_length": length,
              "summary": result.summary(), "checks": checks, "all_passed": all(check["allclose"] for check in checks.values())}
    report_path = args.output.with_suffix(".comparison.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_passed"]:
        raise AssertionError(f"DBC convention comparison failed: {report_path}")
    result.save(args.output)


if __name__ == "__main__":
    main()
