"""Read-only DeepStream MV3DT inspection and explicitly aligned snapshots.

These ROS messages are asynchronous cache snapshots, not source-camera frames.
An alignment manifest is an operator's documented verification, not something
this module can establish from shared frame names or recording endpoints.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
from typing import Any

from atrium_tracking_comparator.deepstream import DeepstreamError, _decode_track_frames


class DeepstreamInputError(ValueError):
    """The artifact or its explicit evaluation alignment is invalid."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeepstreamInputError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise DeepstreamInputError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise DeepstreamInputError(f"{label} must be a finite number")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeepstreamInputError(f"{label} must be a nonempty string")
    return value


def _fields(value: Any, required: set[str], optional: set[str], label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise DeepstreamInputError(f"{label} must be an object")
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing or unknown:
        raise DeepstreamInputError(
            f"{label}: missing fields {sorted(missing)}; unknown fields {sorted(unknown)}"
        )
    return value


def _database(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve()
    if result.is_dir():
        directory = result / "mv3dt_tracks" if (result / "mv3dt_tracks").is_dir() else result
        candidates = sorted(directory.glob("*.db3"))
        if len(candidates) != 1:
            raise DeepstreamInputError(f"Expected exactly one track DB3 in {directory}; found {len(candidates)}")
        result = candidates[0]
    if not result.is_file() or result.suffix != ".db3":
        raise DeepstreamInputError(f"DeepStream input must be an existing .db3 file: {result}")
    # The pure decoder uses SQLite immutable read-only mode. Do not silently
    # ignore a live WAL or recovery journal and evaluate an incomplete database.
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(result) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise DeepstreamInputError(f"Use a completed standalone recording; nonempty SQLite sidecar: {sidecar}")
    return result


def _signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def _read(path: str | Path) -> tuple[list, str, dict]:
    database = _database(path)
    before = _signature(database)
    try:
        snapshots, world_frame = _decode_track_frames(database)
    except (DeepstreamError, OSError) as error:
        raise DeepstreamInputError(f"Cannot read DeepStream recording: {error}") from error
    if before != _signature(database):
        raise DeepstreamInputError("DeepStream database changed during the read")
    if not snapshots:
        raise DeepstreamInputError("DeepStream recording contains no snapshots")
    previous = None
    ids: set[str] = set()
    observations = 0
    empty_count = 0
    for header_ns, _storage_ns, nodes in snapshots:
        if isinstance(header_ns, bool) or not isinstance(header_ns, int) or (previous is not None and header_ns <= previous):
            raise DeepstreamInputError("Snapshot header nanoseconds must be strictly increasing integers")
        previous = header_ns
        seen = set()
        for track_id, x, y, _confidence in nodes:
            if not isinstance(track_id, str) or not track_id:
                raise DeepstreamInputError("Decoded tracker IDs must be nonempty strings")
            if track_id in seen:
                raise DeepstreamInputError(f"Duplicate emitted ID {track_id!r} in one snapshot")
            seen.add(track_id)
            ids.add(track_id)
            _finite(x, "DeepStream x")
            _finite(y, "DeepStream y")
        observations += len(nodes)
        empty_count += not nodes
    metadata = {
        "adapter": "deepstream_async_snapshots",
        "database": str(database),
        "database_signature": before,
        "snapshot_count": len(snapshots),
        "retained_observation_count": observations,
        "empty_snapshot_count": empty_count,
        "distinct_emitted_id_count": len(ids),
        "emitted_track_ids": sorted(ids),
        "world_frame": _text(world_frame, "Decoded world frame"),
        "header_first_ns": str(snapshots[0][0]),
        "header_last_ns": str(snapshots[-1][0]),
        "header_span_s": (snapshots[-1][0] - snapshots[0][0]) / 1e9,
        "track_semantics": "async_cached_tracks",
        "staleness_observable": False,
        "empty_publish_messages_omitted": True,
        "frame_synchronized": False,
        "alignment_available": False,
        "yaw_available": False,
        "yaw_policy": "unavailable: audited bridge identity quaternion is a placeholder, not a heading",
        "ids_preserved": True,
        "limitations": [
            "ROS header time is publish time, not a source-camera or per-person update timestamp.",
            "Cached positions can be stale; per-track cache age is not recorded.",
            "The bridge omits empty publishes; gaps do not prove zero pedestrians.",
            "A shared frame_id alone does not verify historical calibration or sequence correspondence.",
        ],
    }
    return snapshots, world_frame, metadata


def inspect_deepstream(path: str | Path) -> dict:
    """Inspect one recorded DB3 without alignment, scores, video, or cache writes."""
    return _read(path)[2]


def _alignment(value: Mapping | str | Path) -> dict:
    if isinstance(value, (str, Path)):
        try:
            value = json.loads(Path(value).read_text())
        except (OSError, ValueError) as error:
            raise DeepstreamInputError(f"Cannot read alignment manifest: {error}") from error
    value = _fields(value, {"schema_version", "sequence_id", "world_frame", "temporal_alignment", "acknowledge_async_cached_tracks"}, {"absence_policy"}, "alignment")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise DeepstreamInputError("Alignment schema_version must be 1")
    _text(value["sequence_id"], "sequence_id")
    if value["acknowledge_async_cached_tracks"] is not True:
        raise DeepstreamInputError("Explicitly acknowledge asynchronous cached tracks and their unknown staleness")
    world = _fields(value["world_frame"], {"verified", "gt_frame_id", "prediction_frame_id", "coordinate_units", "calibration_evidence"}, set(), "world_frame")
    if world["verified"] is not True:
        raise DeepstreamInputError("A verified common world frame and actual run calibration are required")
    if world["coordinate_units"] != "meters":
        raise DeepstreamInputError("This adapter requires coordinates already expressed in meters")
    if _text(world["gt_frame_id"], "gt_frame_id") != _text(world["prediction_frame_id"], "prediction_frame_id"):
        raise DeepstreamInputError("Coordinates must already share one verified world frame; no spatial transform is inferred")
    _text(world["calibration_evidence"], "calibration_evidence")
    timing = _fields(value["temporal_alignment"], {"verified", "header_origin_ns", "scale", "offset_s", "max_time_delta_s", "evidence"}, set(), "temporal_alignment")
    if timing["verified"] is not True:
        raise DeepstreamInputError("Verified temporal correspondence is required; endpoint/progress alignment is not supported")
    _text(timing["evidence"], "temporal alignment evidence")
    origin = timing["header_origin_ns"]
    if isinstance(origin, str) and re.fullmatch(r"-?[0-9]+", origin):
        origin = int(origin)
    if isinstance(origin, bool) or not isinstance(origin, int):
        raise DeepstreamInputError("header_origin_ns must be exact integer nanoseconds, preferably a decimal string")
    scale = _finite(timing["scale"], "scale")
    offset = _finite(timing["offset_s"], "offset_s")
    tolerance = _finite(timing["max_time_delta_s"], "max_time_delta_s")
    if scale <= 0 or tolerance < 0:
        raise DeepstreamInputError("Temporal scale must be positive and max_time_delta_s nonnegative")
    policy = value.get("absence_policy", "error")
    if policy not in ("error", "empty"):
        raise DeepstreamInputError("absence_policy must be 'error' or explicitly 'empty'")
    return {**value, "world_frame": dict(world), "temporal_alignment": {
        **timing, "header_origin_ns": str(origin), "scale": scale, "offset_s": offset, "max_time_delta_s": tolerance,
    }, "absence_policy": policy}


def load_deepstream_predictions(path: str | Path, gt_frames: Mapping[int, Mapping], alignment: Mapping | str | Path) -> tuple[dict, dict]:
    """Select nearest publish snapshots under an explicit affine alignment.

    ``source_rel_s = scale * ((header_ns - header_origin_ns) / 1e9) + offset_s``.
    GT ``timestamp_s`` must use that same source-relative time axis. Never clamp
    outside snapshot coverage. Ties choose the earlier snapshot. The optional
    explicit ``absence_policy: empty`` creates empty predictions for unmatched
    GT frames and records why; it is an evaluation policy, not observed absence.
    """
    manifest = _alignment(alignment)
    snapshots, world_frame, metadata = _read(path)
    if world_frame != manifest["world_frame"]["prediction_frame_id"]:
        raise DeepstreamInputError("Decoded prediction world frame disagrees with the alignment manifest")
    timing = manifest["temporal_alignment"]
    origin = int(timing["header_origin_ns"])
    times = []
    for header_ns, _, _ in snapshots:
        try:
            value = timing["scale"] * ((header_ns - origin) / 1e9) + timing["offset_s"]
        except OverflowError as error:
            raise DeepstreamInputError("Aligned timestamps exceed finite range") from error
        value = _finite(value, "aligned timestamp")
        if times and value <= times[-1]:
            raise DeepstreamInputError("Affine timestamps lose ordering/precision; use a nearby exact header origin")
        times.append(value)
    if not isinstance(gt_frames, Mapping) or not gt_frames:
        raise DeepstreamInputError("gt_frames must be a nonempty frame-index mapping")
    if any(type(index) is not int or index < 0 for index in gt_frames):
        raise DeepstreamInputError("GT frame indices must be nonnegative integers")
    result, correspondences, unmatched = {}, [], []
    previous_gt_time = None
    for index in sorted(gt_frames):
        frame = gt_frames[index]
        if not isinstance(frame, Mapping) or type(frame.get("frame_index")) is not int or frame["frame_index"] != index:
            raise DeepstreamInputError(f"GT frame {index} has a conflicting frame_index")
        target = _finite(frame.get("timestamp_s"), f"GT frame {index} timestamp_s")
        if previous_gt_time is not None and target <= previous_gt_time:
            raise DeepstreamInputError("GT timestamps must strictly increase with frame index")
        previous_gt_time = target
        reason = None
        chosen = None
        if target < times[0] or target > times[-1]:
            reason = "outside_prediction_coverage"
        else:
            right = bisect_left(times, target)
            candidates = [i for i in (right - 1, right) if 0 <= i < len(times)]
            chosen = min(candidates, key=lambda i: (abs(times[i] - target), i))
            if abs(times[chosen] - target) > timing["max_time_delta_s"]:
                reason = "no_snapshot_within_tolerance"
        if reason:
            entry = {"frame_index": index, "timestamp_s": target, "reason": reason}
            if chosen is not None:
                entry["nearest_delta_s"] = abs(times[chosen] - target)
            if manifest["absence_policy"] == "error":
                raise DeepstreamInputError(f"GT frame {index} ({target}s): {reason}; no endpoint clamping or implicit empty predictions")
            unmatched.append(entry)
            predictions = []
        else:
            header_ns, _, nodes = snapshots[chosen]
            predictions = [{"track_id": track_id, "x": float(x), "y": float(y), "yaw": None} for track_id, x, y, _confidence in nodes]
            correspondences.append({"frame_index": index, "snapshot_index": chosen, "prediction_header_ns": str(header_ns),
                                    "aligned_snapshot_time_s": times[chosen], "delta_s": times[chosen] - target})
        result[index] = {"frame_index": index, "timestamp_s": target, "pred": predictions}
    metadata.update({"alignment_available": True, "alignment": manifest,
                     "alignment_verification": "operator_manifest_assertion",
                     "selection_method": "nearest_async_snapshot_with_explicit_affine_timing",
                     "aligned_coverage_s": [times[0], times[-1]], "requested_frame_count": len(result),
                     "matched_snapshot_frame_count": len(correspondences), "empty_by_policy_frame_count": len(unmatched),
                     "frame_correspondences": correspondences, "unmatched_frames": unmatched,
                     "selected_distinct_snapshot_count": len({row["snapshot_index"] for row in correspondences})})
    return result, metadata
