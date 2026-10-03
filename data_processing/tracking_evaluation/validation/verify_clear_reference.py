"""Differential verification against unchanged, pinned official TrackEval CLEAR.

Run with a Python environment containing NumPy and SciPy. The metric under test
is evaluated using both the optional SciPy and standard-library assignments.
Official code is loaded from the supplied artifact directory without installing
TrackEval or altering its metric/assignment implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time
import types
from unittest.mock import patch

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tracking_evaluation import metrics
from tracking_evaluation.clear_metrics import REFERENCE_COMMIT, aggregate_clear_results, evaluate_clear
from tracking_evaluation.tests.test_clear_metrics import cases

PINNED_CLEAR_SHA256 = "ab3963eac401073fe1b2957c734af1e531368f42fca49627c14422350b2744f0"
COUNT_FIELDS = {"TP": "CLR_TP", "FP": "CLR_FP", "FN": "CLR_FN", "IDSW": "IDSW", "Frag": "Frag"}
FLOAT_FIELDS = {"MOTA": "MOTA", "MOTP_similarity": "MOTP", "similarity_sum": "MOTP_sum"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(official):
    source = official / "trackeval/metrics/clear.py"
    if sha256(source) != PINNED_CLEAR_SHA256:
        raise RuntimeError("Official CLEAR hash does not match pinned source")
    # Compatibility only: no metric code or assignment algorithm is replaced.
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    for name, path in (("reference_trackeval", official / "trackeval"),
                       ("reference_trackeval.metrics", official / "trackeval/metrics")):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    timing = types.ModuleType("reference_trackeval._timing")
    timing.time = lambda function: function
    sys.modules[timing.__name__] = timing
    utils = types.ModuleType("reference_trackeval.utils")
    utils.TrackEvalException = RuntimeError
    utils.init_config = lambda config, defaults, name: {**defaults, **(config or {})}
    sys.modules[utils.__name__] = utils
    return importlib.import_module("reference_trackeval.metrics.clear").CLEAR(
        {"THRESHOLD": .5, "PRINT_CONFIG": False})


def convert(seq, gate):
    """Independent similarity construction, with identical sorted ID order."""
    frames = metrics._frames(seq)
    gids = sorted({n["track_id"] for gt, _ in frames for n in gt})
    pids = sorted({n["track_id"] for _, pred in frames for n in pred})
    gi, pi = {name: i for i, name in enumerate(gids)}, {name: i for i, name in enumerate(pids)}
    data = {"num_gt_ids": len(gids), "num_tracker_ids": len(pids),
            "num_gt_dets": sum(len(gt) for gt, _ in frames),
            "num_tracker_dets": sum(len(pred) for _, pred in frames),
            "num_timesteps": len(frames), "gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    for gt, pred in frames:
        data["gt_ids"].append(np.array([gi[n["track_id"]] for n in gt], dtype=int))
        data["tracker_ids"].append(np.array([pi[n["track_id"]] for n in pred], dtype=int))
        values = np.zeros((len(gt), len(pred)))
        for r, g in enumerate(gt):
            for c, p in enumerate(pred):
                distance = math.hypot(g["x"] - p["x"], g["y"] - p["y"])
                if distance <= gate:
                    values[r, c] = max(0., 1. - distance / (2. * gate))
        data["similarity_scores"].append(values)
    return data


def randomized_cases(count):
    rng = random.Random(20260915)
    result = {}
    for index in range(count):
        frames = []
        for time_index in range(rng.randrange(1, 15)):
            gt = [{"track_id": "g" + str(i), "x": rng.uniform(-2, 2), "y": rng.uniform(-2, 2)}
                  for i in rng.sample(range(6), rng.randrange(7))]
            pred = []
            for i in rng.sample(range(8), rng.randrange(9)):
                anchor = rng.choice(gt) if gt and rng.random() < .75 else {"x": 0., "y": 0.}
                pred.append({"track_id": "p" + str(i), "x": anchor["x"] + rng.uniform(-.6, .6),
                             "y": anchor["y"] + rng.uniform(-.6, .6)})
            frames.append({"frame_index": time_index * 2, "gt": gt, "pred": pred})
        result[f"random_{index:03}"] = {"name": f"random_{index:03}", "frames": frames}
    return result


def compare(ours, reference):
    failures, errors = [], {}
    for left, right in COUNT_FIELDS.items():
        if ours[left] != int(reference[right]):
            failures.append({"field": left, "actual": ours[left], "reference": int(reference[right])})
    for left, right in FLOAT_FIELDS.items():
        error = abs(ours[left] - float(reference[right]))
        errors[left] = error
        tolerance = 1e-9 if left == "similarity_sum" else 1e-12
        if error > tolerance:
            failures.append({"field": left, "actual": ours[left], "reference": float(reference[right]),
                             "absolute_difference": error, "tolerance": tolerance})
    # Distance MOTP is an additional world-distance statistic, not official IoU.
    if ours["TP"]:
        expected_similarity = 1. - ours["MOTP_m"] / (2 * ours["protocol"]["max_distance_m"])
        if abs(ours["MOTP_similarity"] - expected_similarity) > 1e-12:
            failures.append({"field": "MOTP_m", "reason": "linear distance/similarity consistency failed"})
    elif ours["MOTP_m"] is not None:
        failures.append({"field": "MOTP_m", "reason": "expected null with no matches"})
    return failures, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        default=REPO / "investigations/dense_pipeline_eval_20260915/clear_reference")
    parser.add_argument("--sequence", type=Path, action="append", default=[])
    parser.add_argument("--random-count", type=int, default=128)
    args = parser.parse_args()
    start = time.perf_counter()
    official = args.artifact_root / "official"
    reference = load_reference(official)
    sequences = {**cases(), **randomized_cases(args.random_count)}
    sources = []
    real_names = set()
    for path in args.sequence:
        name = "real/" + path.stem
        if name in sequences:
            raise ValueError("duplicate sequence label: " + name)
        sequences[name] = json.loads(path.read_text())
        real_names.add(name)
        sources.append({"label": name, "path": str(path.resolve()), "sha256": sha256(path)})
    official_results = {name: reference.eval_sequence(convert(seq, 1.0)) for name, seq in sequences.items()}
    combined_reference = reference.combine_sequences(official_results)
    checks, failures, max_errors = [], [], {key: 0. for key in FLOAT_FIELDS}
    for backend, solver in (("stdlib_hungarian", None), ("scipy", linear_sum_assignment)):
        results = []
        backend_start = time.perf_counter()
        with patch.object(metrics, "_scipy_assignment", solver):
            for name, seq in sequences.items():
                ours = evaluate_clear(seq)
                problems, errors = compare(ours, official_results[name])
                failures.extend({"sequence": name, "backend": backend, **problem} for problem in problems)
                for field, error in errors.items():
                    max_errors[field] = max(max_errors[field], error)
                results.append(ours)
                if name in real_names:
                    checks.append({"sequence": name, "backend": backend,
                                   "passed": not problems, "frame_count": ours["frame_count"],
                                   "total_GT_observations": ours["total_GT_observations"],
                                   "total_prediction_observations": ours["total_prediction_observations"],
                                   **{key: ours[key] for key in (*COUNT_FIELDS, *FLOAT_FIELDS, "MOTP_m")}})
            combined = aggregate_clear_results(results)
            problems, errors = compare(combined, combined_reference)
            failures.extend({"sequence": "aggregate", "backend": backend, **problem} for problem in problems)
            for field, error in errors.items():
                max_errors[field] = max(max_errors[field], error)
            checks.append({"sequence": "aggregate", "backend": backend, "passed": not problems,
                           "sequence_count": len(results), "elapsed_seconds": time.perf_counter() - backend_start})
    report = {
        "status": "failed" if failures else "passed", "gate_m": 1., "threshold": .5,
        "similarity": "max(0, 1-d/(2*gate)), zero for d>gate",
        "reference_commit": REFERENCE_COMMIT,
        "reference_url": f"https://github.com/JonathonLuiten/TrackEval/blob/{REFERENCE_COMMIT}/trackeval/metrics/clear.py",
        "official_source_files": [{"path": str(path.relative_to(args.artifact_root)), "sha256": sha256(path)}
                                  for path in (official / "trackeval/metrics/clear.py", official / "trackeval/metrics/_base_metric.py", official / "LICENSE")],
        "implementation_files": [{"path": str(path.relative_to(REPO)), "sha256": sha256(path)}
                                 for path in (REPO / "tracking_evaluation/clear_metrics.py", REPO / "tracking_evaluation/metrics.py", Path(__file__))],
        "reference_shims": ["identity timing decorator", "configuration merge without printing", "legacy NumPy dtype aliases"],
        "reference_assignment": "unmodified official SciPy linear_sum_assignment for all comparisons",
        "random_seed": 20260915, "random_sequences": args.random_count, "named_edge_cases": list(cases()),
        "real_inputs": sources, "sequence_comparisons": 2 * len(sequences), "aggregate_comparisons": 2,
        "verified_count_fields": COUNT_FIELDS, "verified_float_fields": FLOAT_FIELDS,
        "maximum_absolute_differences": max_errors, "failures": failures, "checks": checks,
        "notes": ["Whole-frame empty-side branches deliberately retain previous-timestep memory, matching pinned CLEAR.",
                  "Single no-GT or no-prediction sequence MOTA is zero under reference early returns; aggregation recomputes ratios.",
                  "frame_count reports supplied frame coverage even when reference CLR_Frames stays zero in empty-sequence early returns.",
                  "MOTP_m is additional mean world distance; official MOTP is verified as MOTP_similarity.",
                  "Backend tie resolution is not standardized; both backends are compared to the same unchanged reference."],
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "elapsed_seconds": time.perf_counter() - start,
    }
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    destination = args.artifact_root / "verification.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    destination.chmod(0o777)
    print(json.dumps(report, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
