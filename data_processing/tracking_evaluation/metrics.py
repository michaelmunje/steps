"""Tracking metrics for already aligned world-XY observations, in meters.

Independent implementation of the HOTA and global identity definitions in:
https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/hota.py
https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py
https://link.springer.com/article/10.1007/s11263-020-01375-2
Reference for CLEAR conventions:
https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py
Differential verification used TrackEval commit
12c8791b303e0a0b50f753af204249e622d0281a (HOTA/Identity, 2026-09-14).

Protocol: base matches maximize cardinality within an inclusive Euclidean gate,
then minimize distance. ID switches compare each GT's last matched prediction,
including across gaps. Unlike TrackEval CLEAR, matching does not favor the
previous frame's association. MOTA uses the usual count formula; MOTP_m is mean
matched Euclidean error, not mean image similarity. Identity assignment uses
ALL spatially eligible co-occurrences, independently of base frame matches.
These top-level MOTA/IDSW fields are the legacy non-CLEAR base protocol.
The separate ``clear`` result uses pinned TrackEval CLEAR correspondence,
ID-switch and fragmentation rules with a gated world-distance similarity;
its matching does not alter any top-level position, yaw, HOTA or IDF1 metric.

HOTA uses gated world-position similarity: linear max(0, 1-d/scale), or Gaussian
exp(-0.5*(d/scale)^2); both are zero beyond max_distance_m. Global alignment is
formed over the complete sequence before one Hungarian assignment per frame.
The assigned pairs are then filtered at each alpha, as in TrackEval. The default
19 thresholds are 0.05 through 0.95. This is a documented position-similarity
adaptation, not an IoU benchmark score. DetA/AssA/HOTA are arithmetic means over
alphas; HOTA is NOT sqrt(mean(DetA)*mean(AssA)).

All rates are fractions, never percentages. Undefined non-HOTA ratios are None;
HOTA and its submetrics are zero for empty denominators. Circular yaw error is
abs(wrap(pred-gt)) for base matches with both headings known. yaw_coverage divides
that pair count by base TP; unknown headings never become zero-error samples.
Sequence aggregation sums counts/error sums; per-alpha AssA is TP weighted.
Identity namespaces are separate between GT/prediction and between sequences.

Only explicitly supplied frames are scored; no interpolation, cropping, ID
renaming, source filtering, file access, or hidden frame approval occurs here.
Input frame indices must increase strictly. IDs are sorted for repeatable
matrix order; equal-cost optima can differ between solver implementations, so
the chosen assignment backend is recorded. SciPy is optional, with an O(n^2*m)
rectangular Hungarian fallback that uses the same cost matrices.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import math
import sys

from .clear_metrics import aggregate_clear_results, evaluate_clear_frames

try:
    from scipy.optimize import linear_sum_assignment as _scipy_assignment
except ImportError:
    _scipy_assignment = None


_ALPHAS = tuple(index / 20 for index in range(1, 20))
_EPS = sys.float_info.epsilon
_COUNT_FIELDS = (
    "frame_count", "total_GT_observations", "total_prediction_observations",
    "TP", "FP", "FN", "IDSW", "IDTP", "IDFP", "IDFN",
    "yaw_matched_observations", "gt_yaw_observations", "pred_yaw_observations",
)
_SUM_FIELDS = ("distance_sum_m", "yaw_error_sum_rad")
_PROTOCOL = {
    "coordinates": "Euclidean world XY in meters; input frames already aligned",
    "base_matching": "maximum cardinality, then minimum total Euclidean distance; inclusive distance gate",
    "clear_difference": "no previous-frame association preference in assignment",
    "MOTA": "legacy non-CLEAR base MOTA; standard CLEAR correspondence results are in the separate clear object",
    "clear": "separate TrackEval CLEAR correspondence using gated half-height linear world-position similarity",
    "IDSW": "changed prediction ID since the last match to this GT ID, including gaps",
    "IDF1": "global trajectory assignment using all distance-gated co-occurrences",
    "HOTA": "TrackEval global-alignment assignment; gated position similarity; filter assigned pairs per alpha",
    "gaussian": "exp(-0.5*(distance_m/similarity_scale_m)^2)",
    "yaw": "absolute wrapped prediction-minus-GT radians; both headings known on base matches",
    "yaw_coverage": "yaw_matched_observations / TP",
    "aggregation": "sequence-isolated counts/sums; HOTA AssA weighted by per-alpha TP",
    "undefined": "null for zero-denominator non-HOTA metrics; zero for HOTA family",
    "rates": "fractions; MOTA can be negative",
}


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def normalize_config(config=None):
    """Validate protocol settings and return an independent JSON-safe dict."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    allowed = {"max_distance_m", "similarity", "similarity_scale_m", "hota_alphas"}
    if set(config) - allowed:
        raise ValueError("unknown config fields: " + repr(sorted(set(config) - allowed)))
    gate = _finite(config.get("max_distance_m", 1.0), "max_distance_m")
    scale = _finite(config.get("similarity_scale_m", 1.0), "similarity_scale_m")
    if gate <= 0 or scale <= 0:
        raise ValueError("max_distance_m and similarity_scale_m must be positive")
    kind = config.get("similarity", "linear")
    if kind not in ("linear", "gaussian"):
        raise ValueError("similarity must be linear or gaussian")
    alphas = config.get("hota_alphas", _ALPHAS)
    if not isinstance(alphas, (list, tuple)) or not alphas:
        raise ValueError("hota_alphas must be a nonempty ordered list")
    alphas = [_finite(alpha, "HOTA alpha") for alpha in alphas]
    if any(not 0 < alpha <= 1 for alpha in alphas):
        raise ValueError("HOTA alphas must be in (0, 1]")
    if any(first >= second for first, second in zip(alphas, alphas[1:])):
        raise ValueError("HOTA alphas must increase strictly")
    return {"max_distance_m": gate, "similarity": kind,
            "similarity_scale_m": scale, "hota_alphas": alphas}


def _hungarian(cost):
    """Minimum-cost rectangular assignment, with no external dependencies.

    Uses shortest augmenting paths and row/column potentials. Transposing when
    needed means the smaller partition supplies the augmentations. Returns
    min(rows, columns) distinct index pairs; callers apply their own gate.
    """
    if not cost or not cost[0]:
        return []
    rows, columns = len(cost), len(cost[0])
    if any(len(row) != columns for row in cost):
        raise ValueError("assignment matrix must be rectangular")
    if rows > columns:
        transposed = [[cost[row][column] for row in range(rows)] for column in range(columns)]
        return sorted((column, row) for row, column in _hungarian(transposed))
    row_potential = [0.0] * (rows + 1)
    column_potential = [0.0] * (columns + 1)
    owner = [0] * (columns + 1)
    predecessor = [0] * (columns + 1)
    for row in range(1, rows + 1):
        owner[0] = row
        current = 0
        slack = [math.inf] * (columns + 1)
        visited = [False] * (columns + 1)
        while True:
            visited[current] = True
            active_row = owner[current]
            delta, next_column = math.inf, 0
            active_cost = cost[active_row - 1]
            for column in range(1, columns + 1):
                if visited[column]:
                    continue
                reduced = active_cost[column - 1] - row_potential[active_row] - column_potential[column]
                if reduced < slack[column]:
                    slack[column] = reduced
                    predecessor[column] = current
                if slack[column] < delta:
                    delta, next_column = slack[column], column
            for column in range(columns + 1):
                if visited[column]:
                    row_potential[owner[column]] += delta
                    column_potential[column] -= delta
                else:
                    slack[column] -= delta
            current = next_column
            if owner[current] == 0:
                break
        while current:
            previous = predecessor[current]
            owner[current] = owner[previous]
            current = previous
    return sorted((owner[column] - 1, column - 1)
                  for column in range(1, columns + 1) if owner[column])


def _assignment(cost):
    if not cost or not cost[0]:
        return []
    if _scipy_assignment is not None:
        rows, columns = _scipy_assignment(cost)
        return sorted((int(row), int(column)) for row, column in zip(rows, columns))
    return _hungarian(cost)


def _base_matches(distances, gate):
    if not distances or not distances[0]:
        return []
    # A forbidden edge costs more than ALL possible valid-distance costs.
    # Every partial matching extends to a full rectangular assignment, so this
    # is exactly cardinality first, distance second, without greedy filtering.
    forbidden = min(len(distances), len(distances[0])) + 1.0
    cost = [[distance / gate if distance <= gate else forbidden for distance in row]
            for row in distances]
    return [(row, column) for row, column in _assignment(cost)
            if distances[row][column] <= gate]


def _observations(values, label):
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    seen, normalized = set(), []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} observations must be mappings")
        identity = value.get("track_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{label} track_id must be a nonempty string")
        if identity in seen:
            raise ValueError(f"duplicate {label} track_id {identity!r} in one frame")
        seen.add(identity)
        yaw = value.get("yaw")
        normalized.append({"track_id": identity,
                           "x": _finite(value.get("x"), f"{label} x"),
                           "y": _finite(value.get("y"), f"{label} y"),
                           "yaw": None if yaw is None else math.remainder(_finite(yaw, f"{label} yaw"), math.tau)})
    return sorted(normalized, key=lambda node: node["track_id"])


def _frames(sequence):
    if not isinstance(sequence, Mapping) or not isinstance(sequence.get("name"), str) or not sequence["name"]:
        raise ValueError("sequence requires a nonempty name")
    if not isinstance(sequence.get("frames"), list):
        raise ValueError("sequence frames must be a list")
    previous_index, previous_time = -1, None
    result = []
    for frame in sequence["frames"]:
        if not isinstance(frame, Mapping):
            raise ValueError("frame must be a mapping")
        index = frame.get("frame_index")
        if isinstance(index, bool) or not isinstance(index, int) or index <= previous_index:
            raise ValueError("frame_index must be nonnegative and strictly increasing")
        timestamp = frame.get("timestamp_s")
        if timestamp is not None:
            timestamp = _finite(timestamp, "timestamp_s")
            if previous_time is not None and timestamp < previous_time:
                raise ValueError("known timestamps must not decrease")
            previous_time = timestamp
        previous_index = index
        result.append((_observations(frame.get("gt"), "GT"),
                       _observations(frame.get("pred"), "prediction")))
    return result


def _similarity(distance, config):
    if distance > config["max_distance_m"]:
        return 0.0
    ratio = distance / config["similarity_scale_m"]
    if config["similarity"] == "linear":
        return max(0.0, 1.0 - ratio)
    return math.exp(-0.5 * ratio * ratio)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _finish(result):
    """Derive all non-additive fields only after sufficient statistics exist."""
    tp, fp, fn = result["TP"], result["FP"], result["FN"]
    result.update(matched_observations=tp, precision=_ratio(tp, tp + fp),
                  recall=_ratio(tp, tp + fn), MOTP_m=_ratio(result["distance_sum_m"], tp),
                  MOTA=(1 - (fn + fp + result["IDSW"]) / result["total_GT_observations"]
                        if result["total_GT_observations"] else None),
                  IDF1=_ratio(2 * result["IDTP"], 2 * result["IDTP"] + result["IDFP"] + result["IDFN"]),
                  yaw_MAE_rad=_ratio(result["yaw_error_sum_rad"], result["yaw_matched_observations"]),
                  yaw_coverage=_ratio(result["yaw_matched_observations"], tp))
    result["yaw_MAE_deg"] = (math.degrees(result["yaw_MAE_rad"])
                             if result["yaw_MAE_rad"] is not None else None)
    for row in result["hota_per_alpha"]:
        row["AssA"] = row["association_sum"] / max(1, row["HOTA_TP"])
        row["DetA"] = row["HOTA_TP"] / max(1, row["HOTA_TP"] + row["HOTA_FP"] + row["HOTA_FN"])
        row["HOTA"] = math.sqrt(row["AssA"] * row["DetA"])
    for metric in ("HOTA", "DetA", "AssA"):
        result[metric] = math.fsum(row[metric] for row in result["hota_per_alpha"]) / len(result["hota_per_alpha"])
    return result


def evaluate_sequence(sequence, config=None):
    """Score one already aligned sequence, without modifying the input.

    sequence = {"name": str, "frames": [{"frame_index": int,
        "timestamp_s": float | None, "gt": [...], "pred": [...]}]}
    Each observation supplies track_id (string), finite x/y, and optional yaw
    in radians (None means unknown). Duplicate IDs within either side/frame,
    invalid numbers, and ambiguous frame ordering are rejected.
    """
    config = normalize_config(config)
    frames = _frames(sequence)
    gt_counts = Counter(node["track_id"] for gt, _ in frames for node in gt)
    pred_counts = Counter(node["track_id"] for _, pred in frames for node in pred)
    result = {field: 0 for field in _COUNT_FIELDS}
    result.update({field: 0.0 for field in _SUM_FIELDS})
    result.update(schema_version=1, name=sequence["name"], sequence_count=1,
                  config=config, protocol=dict(_PROTOCOL), frame_count=len(frames),
                  assignment_backend="scipy" if _scipy_assignment is not None else "stdlib_hungarian",
                  total_GT_observations=sum(gt_counts.values()),
                  total_prediction_observations=sum(pred_counts.values()))
    eligible_counts = Counter()
    soft_counts = defaultdict(float)
    last_prediction = {}
    cached = []
    distances_to_sum, angles_to_sum = [], []
    for gt, pred in frames:
        result["gt_yaw_observations"] += sum(node["yaw"] is not None for node in gt)
        result["pred_yaw_observations"] += sum(node["yaw"] is not None for node in pred)
        distances = [[math.hypot(g["x"] - p["x"], g["y"] - p["y"]) for p in pred] for g in gt]
        similarities = [[_similarity(distance, config) for distance in row] for row in distances]
        cached.append((gt, pred, similarities))
        matches = _base_matches(distances, config["max_distance_m"])
        result["TP"] += len(matches)
        result["FP"] += len(pred) - len(matches)
        result["FN"] += len(gt) - len(matches)
        for row, column in matches:
            g, p = gt[row], pred[column]
            gid, pid = g["track_id"], p["track_id"]
            if gid in last_prediction and last_prediction[gid] != pid:
                result["IDSW"] += 1
            last_prediction[gid] = pid
            distances_to_sum.append(distances[row][column])
            if g["yaw"] is not None and p["yaw"] is not None:
                angles_to_sum.append(abs(math.remainder(p["yaw"] - g["yaw"], math.tau)))
        row_sums = [math.fsum(row) for row in similarities]
        column_sums = [math.fsum(similarities[row][column] for row in range(len(gt)))
                       for column in range(len(pred))]
        for row, g in enumerate(gt):
            for column, p in enumerate(pred):
                pair = (g["track_id"], p["track_id"])
                if distances[row][column] <= config["max_distance_m"]:
                    eligible_counts[pair] += 1
                similarity = similarities[row][column]
                denominator = row_sums[row] + column_sums[column] - similarity
                if denominator > _EPS:
                    soft_counts[pair] += similarity / denominator
    result["distance_sum_m"] = math.fsum(distances_to_sum)
    result["yaw_error_sum_rad"] = math.fsum(angles_to_sum)
    result["yaw_matched_observations"] = len(angles_to_sum)

    # Matching trajectories maximizes eligible pair counts. The unassigned
    # trajectory FP/FN costs are constant totals minus twice this objective;
    # zero-weight completion is equivalent to explicit unmatched dummy nodes.
    gt_ids, pred_ids = sorted(gt_counts), sorted(pred_counts)
    identity_cost = [[-eligible_counts[(gid, pid)] for pid in pred_ids] for gid in gt_ids]
    result["IDTP"] = sum(eligible_counts[(gt_ids[row], pred_ids[column])]
                         for row, column in _assignment(identity_cost))
    result["IDFN"] = result["total_GT_observations"] - result["IDTP"]
    result["IDFP"] = result["total_prediction_observations"] - result["IDTP"]

    alignment = {pair: value / (gt_counts[pair[0]] + pred_counts[pair[1]] - value)
                 for pair, value in soft_counts.items()}
    alpha_matches = [Counter() for _ in config["hota_alphas"]]
    for gt, pred, similarities in cached:
        costs = [[-alignment.get((g["track_id"], p["track_id"]), 0.0) * similarities[row][column]
                  for column, p in enumerate(pred)] for row, g in enumerate(gt)]
        matches = _assignment(costs)
        for alpha, counts in zip(config["hota_alphas"], alpha_matches):
            for row, column in matches:
                if similarities[row][column] > 0 and similarities[row][column] >= alpha - _EPS:
                    counts[(gt[row]["track_id"], pred[column]["track_id"])] += 1
    result["hota_per_alpha"] = []
    for alpha, counts in zip(config["hota_alphas"], alpha_matches):
        tp = sum(counts.values())
        association_sum = math.fsum(count * count / (gt_counts[gid] + pred_counts[pid] - count)
                                    for (gid, pid), count in counts.items())
        result["hota_per_alpha"].append({"alpha": alpha, "HOTA_TP": tp,
            "HOTA_FN": result["total_GT_observations"] - tp,
            "HOTA_FP": result["total_prediction_observations"] - tp,
            "association_sum": association_sum})
    result["clear"] = evaluate_clear_frames(frames, config["max_distance_m"], _assignment,
                                             result["assignment_backend"])
    return _finish(result)


def aggregate_results(results):
    """Combine sequence results by sufficient statistics, never frame averages.

    Configurations must agree. IDs from different sequences are never globally
    matched to one another. An empty list yields empty default-protocol scores.
    """
    results = list(results)
    if not results:
        empty = evaluate_sequence({"name": "aggregate", "frames": []})
        empty["sequence_count"] = 0
        empty["sequence_names"] = []
        empty["clear"]["sequence_count"] = 0
        return empty
    config = normalize_config(results[0]["config"])
    if any(normalize_config(item["config"]) != config for item in results):
        raise ValueError("cannot aggregate results with different metric configurations")
    if any(item.get("schema_version") != 1 or item.get("protocol") != _PROTOCOL for item in results):
        raise ValueError("cannot aggregate results from a different metric protocol")
    if any(not isinstance(item.get("clear"), Mapping) for item in results):
        raise ValueError("cannot aggregate a result missing its separate CLEAR calculation")
    result = {field: sum(item[field] for item in results) for field in _COUNT_FIELDS}
    result.update({field: math.fsum(item[field] for item in results) for field in _SUM_FIELDS})
    backends = sorted({item["assignment_backend"] for item in results})
    result.update(schema_version=1, name="aggregate", config=config, protocol=dict(_PROTOCOL),
                  sequence_count=sum(item["sequence_count"] for item in results),
                  sequence_names=[name for item in results for name in item.get("sequence_names", [item["name"]])],
                  assignment_backend=backends[0] if len(backends) == 1 else "mixed:" + ",".join(backends))
    result["hota_per_alpha"] = []
    for index, alpha in enumerate(config["hota_alphas"]):
        rows = [item["hota_per_alpha"][index] for item in results]
        if any(row["alpha"] != alpha for row in rows):
            raise ValueError("HOTA alpha rows disagree with configuration")
        result["hota_per_alpha"].append({"alpha": alpha,
            **{field: sum(row[field] for row in rows) for field in ("HOTA_TP", "HOTA_FP", "HOTA_FN")},
            "association_sum": math.fsum(row["association_sum"] for row in rows)})
    result["clear"] = aggregate_clear_results(item["clear"] for item in results)
    return _finish(result)
