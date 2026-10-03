"""Compare saved dense production results with pinned official HOTA/Identity.

This verifier loads normalized sequences and existing result JSONs. It does not
rerun predictions or modify scores. Official matching uses SciPy unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tracking_evaluation import metrics
from tracking_evaluation.validation.verify_clear_reference import load_reference, sha256


def convert(sequence, config):
    frames = metrics._frames(sequence)
    gids = sorted({n["track_id"] for gt, _ in frames for n in gt})
    pids = sorted({n["track_id"] for _, pred in frames for n in pred})
    gi, pi = {name: i for i, name in enumerate(gids)}, {name: i for i, name in enumerate(pids)}
    data = {"num_gt_ids": len(gids), "num_tracker_ids": len(pids),
            "num_gt_dets": sum(len(gt) for gt, _ in frames),
            "num_tracker_dets": sum(len(pred) for _, pred in frames),
            "num_timesteps": len(frames), "gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    binary = []
    for gt, pred in frames:
        data["gt_ids"].append(np.array([gi[n["track_id"]] for n in gt], dtype=int))
        data["tracker_ids"].append(np.array([pi[n["track_id"]] for n in pred], dtype=int))
        similarity, eligible = np.zeros((len(gt), len(pred))), np.zeros((len(gt), len(pred)))
        for r, g in enumerate(gt):
            for c, p in enumerate(pred):
                distance = math.hypot(g["x"] - p["x"], g["y"] - p["y"])
                if distance <= config["max_distance_m"]:
                    eligible[r, c] = 1
                    relative = distance / config["similarity_scale_m"]
                    similarity[r, c] = max(0., 1. - relative) if config["similarity"] == "linear" else math.exp(-.5 * relative * relative)
        data["similarity_scores"].append(similarity)
        binary.append(eligible)
    return data, {**data, "similarity_scores": binary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=REPO / "investigations/dense_pipeline_eval_20260915")
    parser.add_argument("--names", nargs="+", default=["sam3_cleaned", "sam3_cleaned_observed", "deepstream", "deepstream_independent"])
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.work_root
    official = root / "clear_reference/official"
    load_reference(official)  # Shared namespace/dependency shims; verifies pinned CLEAR.
    provenance = json.loads((official / "hota_identity_provenance.json").read_text())
    for path, expected in provenance["files"].items():
        if sha256(official / path) != expected["sha256"]:
            raise RuntimeError("Pinned reference source changed: " + path)
    HOTA = importlib.import_module("reference_trackeval.metrics.hota").HOTA
    Identity = importlib.import_module("reference_trackeval.metrics.identity").Identity
    checks, failures, maximum_difference = [], [], 0.
    for name in args.names:
        sequence_path, report_path = root / f"sequences/{name}.json", root / f"runs/{name}/report.json"
        sequence, report = json.loads(sequence_path.read_text()), json.loads(report_path.read_text())
        config = metrics.normalize_config(report["metric_config"])
        if config["hota_alphas"] != [i / 20 for i in range(1, 20)]:
            raise ValueError("Official default HOTA alpha list differs from saved report")
        data, binary = convert(sequence, config)
        ref_hota = HOTA().eval_sequence(data)
        ref_id = Identity({"THRESHOLD": .5, "PRINT_CONFIG": False}).eval_sequence(binary)
        comparisons = 0
        before = len(failures)
        for output_name, ours in (("sequence", report["sequences"][0]["metrics"]), ("combined", report["combined"])):
            for field in ("HOTA_TP", "HOTA_FP", "HOTA_FN", "HOTA", "DetA", "AssA"):
                actual = np.array([row[field] for row in ours["hota_per_alpha"]])
                expected = ref_hota[field]
                error = float(np.max(np.abs(actual - expected)))
                maximum_difference = max(maximum_difference, error)
                if error > 1e-12:
                    failures.append({"sequence": name, "output": output_name, "field": field, "maximum_absolute_difference": error})
                comparisons += len(actual)
            for field in ("HOTA", "DetA", "AssA", "IDTP", "IDFP", "IDFN", "IDF1"):
                expected = float(np.mean(ref_hota[field])) if field in ref_hota else float(ref_id[field])
                error = abs(ours[field] - expected)
                maximum_difference = max(maximum_difference, error)
                if error > 1e-12:
                    failures.append({"sequence": name, "output": output_name, "field": field,
                                     "actual": ours[field], "reference": expected})
                comparisons += 1
            for field, expected in (("frame_count", data["num_timesteps"]), ("total_GT_observations", data["num_gt_dets"]),
                                    ("total_prediction_observations", data["num_tracker_dets"])):
                if ours[field] != expected:
                    failures.append({"sequence": name, "output": output_name, "field": field,
                                     "actual": ours[field], "expected": expected})
        checks.append({"sequence": name, "passed": len(failures) == before, "scalar_comparisons": comparisons,
                       "sequence_sha256": sha256(sequence_path), "report_sha256": sha256(report_path),
                       "production_assignment_backend": report["combined"]["assignment_backend"],
                       "HOTA": report["combined"]["HOTA"], "IDF1": report["combined"]["IDF1"]})
    result = {"status": "failed" if failures else "passed", "checks": checks, "failures": failures,
              "maximum_absolute_difference": maximum_difference, "reference": provenance,
              "reference_assignment": "unchanged official SciPy assignment",
              "input": "normalized sequence snapshots and saved production reports; no model inference",
              "HOTA_similarity": "saved metric config; default linear 1-d/1m, gated beyond 1m",
              "Identity_similarity": "binary inclusive distance<=1m; official threshold0.5",
              "reference_shims": ["identity timing decorator", "default/config merge", "legacy NumPy dtype aliases"],
              "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
              "elapsed_seconds": time.perf_counter() - started,
              "verification_script_sha256": sha256(Path(__file__))}
    destination = root / "clear_reference/hota_identity_dense_verification.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    destination.chmod(0o777)
    print(json.dumps(result, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
