"""Sequence-isolated CLEAR bookkeeping on gated world-position similarity.

The correspondence and fragmentation rules follow TrackEval CLEAR at commit
12c8791b303e0a0b50f753af204249e622d0281a, independently of this repository's
cardinality-first base position/yaw matches and its HOTA/identity assignments.
This module does not interpret world distances as image IoU.
"""
from __future__ import annotations

from collections import Counter
import math
import sys


REFERENCE_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
_EPS = sys.float_info.epsilon
_THRESHOLD = 0.5
_CONTINUITY_BONUS = 1000.0
_COUNTS = ("frame_count", "total_GT_observations", "total_prediction_observations",
           "TP", "FP", "FN", "IDSW", "Frag")


def protocol(gate):
    return {
        "reference": "TrackEval CLEAR",
        "reference_commit": REFERENCE_COMMIT,
        "coordinates": "world-ground XY in metres; not image IoU",
        "max_distance_m": gate,
        "similarity": "max(0, 1 - distance_m / (2 * max_distance_m)); zero when distance_m > max_distance_m",
        "threshold": _THRESHOLD,
        "continuity_bonus": _CONTINUITY_BONUS,
        "assignment": "maximize sum(1000 * previous_timestep_same_ID + similarity); ineligible scores zeroed before Hungarian assignment",
        "eligibility": "similarity >= threshold - machine epsilon; selected score > machine epsilon",
        "IDSW": "prediction ID changed since this GT's last successful match, including gaps",
        "Frag": "sum over GT identities of tracking-segment starts minus one, following pinned reference bookkeeping",
        "empty_frame_bookkeeping": "when either side has no observations, accumulate FP/FN and retain previous-timestep state, matching the pinned reference",
        "empty_sequence_MOTA": "zero for a single sequence with no GT or no predictions, matching reference early returns; aggregation recomputes from summed counts",
        "MOTP_m": "mean distance over CLEAR matches; null if no CLEAR matches",
        "MOTP_similarity": "mean CLEAR similarity over CLEAR matches; zero if no CLEAR matches",
        "aggregation": "sum counts/errors with identity history isolated per sequence; recompute ratios",
    }


def position_similarity(distance, gate):
    """A half-height linear similarity keeps exactly the inclusive XY gate."""
    return 0.0 if distance > gate else max(0.0, 1.0 - distance / (2.0 * gate))


def _finish(result, *, single_sequence=False):
    gt = result["TP"] + result["FN"]
    if single_sequence and (not gt or not result["total_prediction_observations"]):
        result["MOTA"] = 0.0
    else:
        result["MOTA"] = (result["TP"] - result["FP"] - result["IDSW"]) / max(1, gt)
    result["MOTP_m"] = result["distance_sum_m"] / result["TP"] if result["TP"] else None
    result["MOTP_similarity"] = result["similarity_sum"] / max(1, result["TP"])
    return result


def evaluate_clear_frames(frames, gate, assignment, assignment_backend):
    """Score the already validated normalized frame pairs from metrics._frames.

    ``assignment`` minimizes a rectangular cost matrix. Passing the parent
    evaluator's function preserves its recorded optional-SciPy/fallback choice.
    Reference empty-frame branches deliberately do not clear continuity state.
    """
    result = {key: 0 for key in _COUNTS}
    result.update(schema_version=1, sequence_count=1, frame_count=len(frames),
                  total_GT_observations=sum(len(gt) for gt, _ in frames),
                  total_prediction_observations=sum(len(pred) for _, pred in frames),
                  distance_sum_m=0.0, similarity_sum=0.0,
                  protocol=protocol(gate), assignment_backend=assignment_backend)
    last_match, previous_timestep = {}, {}
    segments = Counter()
    distances_matched, similarities_matched = [], []
    for gt, pred in frames:
        if not gt:
            result["FP"] += len(pred)
            continue
        if not pred:
            result["FN"] += len(gt)
            continue
        distances = [[math.hypot(g["x"] - p["x"], g["y"] - p["y"])
                      for p in pred] for g in gt]
        similarities = [[position_similarity(d, gate) for d in row] for row in distances]
        scores = [[(_CONTINUITY_BONUS * (previous_timestep.get(g["track_id"]) == p["track_id"]) + similarities[r][c])
                   if similarities[r][c] >= _THRESHOLD - _EPS else 0.0
                   for c, p in enumerate(pred)] for r, g in enumerate(gt)]
        matches = [(r, c) for r, c in assignment([[-score for score in row] for row in scores])
                   if scores[r][c] > _EPS]
        now = {}
        for r, c in matches:
            gid, pid = gt[r]["track_id"], pred[c]["track_id"]
            if gid in last_match and last_match[gid] != pid:
                result["IDSW"] += 1
            if gid not in previous_timestep:
                segments[gid] += 1
            last_match[gid] = pid
            now[gid] = pid
            distances_matched.append(distances[r][c])
            similarities_matched.append(similarities[r][c])
        previous_timestep = now
        result["TP"] += len(matches)
        result["FP"] += len(pred) - len(matches)
        result["FN"] += len(gt) - len(matches)
    result["Frag"] = sum(count - 1 for count in segments.values())
    result["distance_sum_m"] = math.fsum(distances_matched)
    result["similarity_sum"] = math.fsum(similarities_matched)
    return _finish(result, single_sequence=True)


def evaluate_clear(sequence, config=None):
    """Public CLEAR-only entry point with normal evaluator input validation."""
    from . import metrics
    config = metrics.normalize_config(config)
    backend = "scipy" if metrics._scipy_assignment is not None else "stdlib_hungarian"
    return evaluate_clear_frames(metrics._frames(sequence), config["max_distance_m"],
                                 metrics._assignment, backend)


def aggregate_clear_results(results):
    results = list(results)
    if not results:
        raise ValueError("CLEAR aggregation requires at least one result")
    first_protocol = results[0].get("protocol")
    if any(item.get("schema_version") != 1 or item.get("protocol") != first_protocol for item in results):
        raise ValueError("cannot aggregate different CLEAR protocols")
    result = {key: sum(item[key] for item in results) for key in _COUNTS}
    backends = sorted({item["assignment_backend"] for item in results})
    result.update(schema_version=1, protocol=dict(first_protocol),
                  sequence_count=sum(item["sequence_count"] for item in results),
                  assignment_backend=backends[0] if len(backends) == 1 else "mixed:" + ",".join(backends),
                  distance_sum_m=math.fsum(item["distance_sum_m"] for item in results),
                  similarity_sum=math.fsum(item["similarity_sum"] for item in results))
    return _finish(result)
