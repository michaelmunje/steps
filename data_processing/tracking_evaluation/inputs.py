"""Read-only adapters for reviewed GT exports and saved SAM3D predictions.

Both loaders return ``({index: normalized_frame}, metadata)``. A normalized
frame has frame_index, timestamp_s and gt or pred; people have track_id, x, y,
and yaw. No annotation database, working archive, or prediction ID remapping
is involved. Only the requested prefix of a SAM3D JSONL is read.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any


class InputError(ValueError):
    """An input cannot be evaluated without guessing or discarding bad data."""


_FPS = 20.0
_POLICIES = {"annotator-observed", "all-emitted"}
_COORDINATES = {"plane": "BEV", "reference": "gdc_atrium", "units": "metres"}
_ORIENTATION = {"units": "radians", "zero": "+X", "positive": "counterclockwise toward +Y"}


def _number(value: Any, context: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise InputError(f"{context} must be finite numeric" + (" or null" if nullable else ""))
    return float(value)


def _index(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputError(f"{context} must be a nonnegative integer")
    return value


def _timestamp_ns(value: Any, context: str) -> int:
    # JSON exports encode nanoseconds as decimal strings to avoid JavaScript
    # precision loss; original JSONL records use integers. Never round a float.
    if isinstance(value, str) and value and value.isascii() and value.isdecimal():
        value = int(value)
    return _index(value, context)


def _id(value: Any, context: str) -> str:
    # Match the source adapter's integer-ID serialization. Strings are never
    # parsed, stripped, renumbered, or compared against the other namespace.
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        if isinstance(value, int) or (math.isfinite(value) and value.is_integer()):
            return str(int(value))
    raise InputError(f"{context} must be a nonempty string or integral numeric ID")


def _object(value: Any, context: str) -> dict:
    if not isinstance(value, dict):
        raise InputError(f"{context} must be an object")
    return value


def _array(value: Any, context: str) -> list:
    if not isinstance(value, list):
        raise InputError(f"{context} must be an array")
    return value


def _bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise InputError(f"{context} must be boolean")
    return value


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as stream:
            return _object(json.load(stream), str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InputError(f"Cannot read JSON {path}: {error}") from error


def _signature(path: Path) -> dict:
    try:
        if not path.is_file():
            raise InputError(f"Required input file is missing: {path}")
        stat = path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError as error:
        raise InputError(f"Cannot stat {path}: {error}") from error


def _person(raw: Any, context: str, *, gt_export: bool = False) -> dict:
    person = _object(raw, context)
    return {
        "track_id": _id(person.get("gt_id" if gt_export else "track_id"), context + ".track_id"),
        "x": _number(person.get("x"), context + ".x"),
        "y": _number(person.get("y"), context + ".y"),
        "yaw": _number(person.get("yaw_rad" if gt_export else "yaw"), context + ".yaw", nullable=True),
    }


def _people(raw: Any, context: str, *, gt_export: bool = False) -> list[dict]:
    people, seen = [], set()
    for position, item in enumerate(_array(raw, context)):
        item = _object(item, f"{context}[{position}]")
        if gt_export and "extra" in item and _bool(item["extra"], context + ".extra"):
            continue
        person = _person(item, f"{context}[{position}]", gt_export=gt_export)
        if person["track_id"] in seen:
            raise InputError(f"{context} contains duplicate track ID {person['track_id']!r}")
        seen.add(person["track_id"])
        people.append(person)
    return people


def _frame(raw: Any, side: str, *, gt_export: bool = False) -> dict:
    raw = _object(raw, "frame")
    index = _index(raw.get("frame_index"), "frame.frame_index")
    context = f"frame {index}"
    if gt_export and raw.get("reviewed") is not True:
        raise InputError(f"{context} is not reviewed; use ground_truth.json, not a working archive")
    result = {
        "frame_index": index,
        "timestamp_s": _number(raw.get("time_s" if gt_export else "timestamp_s"),
                               context + ".timestamp_s", nullable=True),
        side: _people(raw.get("ground_truth" if gt_export else side), context + "." + side,
                      gt_export=gt_export),
    }
    if "source_timestamp_ns" in raw:
        result["source_timestamp_ns"] = _timestamp_ns(raw["source_timestamp_ns"], context + ".source_timestamp_ns")
    return result


def _normalized_frames(document: dict, side: str, *, gt_export: bool = False) -> dict[int, dict]:
    result = {}
    for raw in _array(document.get("frames"), "frames"):
        frame = _frame(raw, side, gt_export=gt_export)
        index = frame["frame_index"]
        if index in result:
            raise InputError(f"Duplicate frame_index {index}")
        result[index] = frame
    return dict(sorted(result.items()))


def _document_metadata(document: dict, path: Path, format_name: str) -> dict:
    session = _object(document.get("session", {}), "session")
    extra = _object(document.get("metadata", {}), "metadata")
    fingerprint = session.get("source_fingerprint", document.get("source_fingerprint", extra.get("source_fingerprint")))
    if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint):
        raise InputError("source_fingerprint must be a nonempty string")
    fps = document.get("fps", extra.get("fps"))
    if fps is not None and (_number(fps, "fps") or 0) <= 0:
        raise InputError("fps must be positive")
    result = {
        "format": format_name, "path": str(path), "name": session.get("name", document.get("name", path.stem)),
        "session": session, "source_fingerprint": fingerprint,
        "fps": fps, "coordinate_system": document.get("coordinate_system", extra.get("coordinate_system")),
        "orientation_convention": document.get("orientation_convention", extra.get("orientation_convention")),
        "warnings": [],
    }
    coordinates, orientation = result["coordinate_system"], result["orientation_convention"]
    if coordinates is not None:
        coordinates = _object(coordinates, "coordinate_system")
        if coordinates.get("units") not in {"metres", "meters", "metre", "meter", "m"}:
            raise InputError("coordinate_system.units must be metres/meters")
        if "plane" in coordinates and coordinates["plane"] not in {"BEV", "ground", "ground-plane"}:
            raise InputError("coordinate_system.plane must be a ground/BEV plane")
    else:
        result["warnings"].append("Coordinate convention unspecified: caller must assert aligned ground-plane metres.")
    if orientation is not None:
        orientation = _object(orientation, "orientation_convention")
        if orientation.get("units") not in {"radians", "radian", "rad"}:
            raise InputError("orientation_convention.units must be radians")
        if orientation.get("zero") != "+X":
            raise InputError("orientation_convention.zero must be +X")
        if orientation.get("positive") not in {"counterclockwise toward +Y", "counterclockwise", "CCW"}:
            raise InputError("orientation_convention.positive must be counterclockwise toward +Y")
    else:
        result["warnings"].append("Orientation convention unspecified: caller must assert radians, zero +X, positive counterclockwise toward +Y.")
    return result


def load_ground_truth(path: str | Path) -> tuple[dict[int, dict], dict]:
    """Load a reviewed ``ground_truth.json`` or portable normalized GT JSON.

    For an export, ``ground_truth`` is authoritative, including manual people.
    Per-person review flags and source correspondences do not filter GT. Empty
    frames survive; an export with no eligible frames is an actionable error.
    """
    path = Path(path).expanduser().absolute()
    if path.is_dir():
        path = path / "ground_truth.json"
    before = _signature(path)
    document = _read_json(path)
    raw_frames = _array(document.get("frames"), "frames")
    exported = "session" in document or any(isinstance(f, dict) and "ground_truth" in f for f in raw_frames)
    if not raw_frames:
        raise InputError("Ground truth contains zero eligible frames; no approvals or snapshots will be created")
    frames = _normalized_frames(document, "gt", gt_export=exported)
    metadata = _document_metadata(document, path, "reviewed-ground-truth" if exported else "normalized")
    metadata.update(frame_count=len(frames), file_signature=before)
    if exported:
        metadata["fps"] = _FPS
        metadata["warnings"].append("Reviewed export may have coverage gaps; only its explicit frames are available.")
    else:
        metadata["warnings"].append("Normalized GT is caller-supplied; annotation review status is not independently certified.")
    if _signature(path) != before:
        raise InputError("Ground truth file changed during reading")
    return frames, metadata


def _sam3_person(raw: dict, context: str, *, fused: bool = False) -> dict:
    xy = raw.get("position_smoothed_xy") if fused else None
    if xy is None:
        xy = raw.get("position_ground_xy")
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        raise InputError(f"{context}.position_ground_xy must contain [x, y]")
    return {
        "track_id": _id(raw.get("track_id"), context + ".track_id"),
        "x": _number(xy[0], context + ".x"), "y": _number(xy[1], context + ".y"),
        "yaw": _number(raw.get("yaw_ground_rad"), context + ".yaw_ground_rad", nullable=True),
    }


def _sam3_frame(raw: dict, policy: str, counts: dict) -> dict:
    index = _index(raw.get("frame_index"), "frame_index")
    stamp = _index(raw.get("timestamp_ns"), f"frame {index}.timestamp_ns")
    tracks, seen, tracked_ids, fused_ids = [], set(), set(), set()
    for offset, value in enumerate(_array(raw.get("tracked_people"), f"frame {index}.tracked_people")):
        context = f"frame {index}.tracked_people[{offset}]"
        person = _object(value, context)
        ident = _id(person.get("track_id"), context + ".track_id")
        if ident in tracked_ids:
            raise InputError(f"frame {index} contains duplicate tracked ID {ident!r}")
        tracked_ids.add(ident)
        observed = _bool(person.get("observed_this_frame"), context + ".observed_this_frame")
        stale = _bool(person.get("stale", False), context + ".stale")
        counts["tracked_records"] += 1
        if policy == "annotator-observed" and not observed:
            counts["excluded_unobserved_tracked"] += 1
            if stale:
                counts["excluded_unobserved_stale_tracked"] += 1
            continue
        if observed and stale:
            raise InputError(f"{context} is simultaneously observed and stale")
        tracks.append(_sam3_person(person, context))
        seen.add(ident)
        counts["emitted_tracked"] += 1
    for offset, value in enumerate(_array(raw.get("fused_people", []), f"frame {index}.fused_people")):
        context = f"frame {index}.fused_people[{offset}]"
        person = _object(value, context)
        counts["fused_records"] += 1
        retroactive = _bool(person.get("global_birth_retroactively_labeled", False), context + ".retroactive")
        if not retroactive:
            counts["excluded_nonretroactive_fused"] += 1
            continue
        if person.get("track_id") is None:
            counts["excluded_unidentified_retroactive_fused"] += 1
            continue
        ident = _id(person["track_id"], context + ".track_id")
        if ident in fused_ids:
            raise InputError(f"frame {index} contains duplicate finalized fused ID {ident!r}")
        fused_ids.add(ident)
        if ident in seen:
            counts["excluded_fused_already_emitted"] += 1
            continue
        tracks.append(_sam3_person(person, context, fused=True))
        seen.add(ident)
        counts["emitted_retroactive_fused"] += 1
    return {"frame_index": index, "timestamp_s": index / _FPS,
            "source_timestamp_ns": stamp, "pred": tracks}


def _source_identity(directory: Path, source_file: Path) -> tuple[str | None, dict, dict]:
    # Exact comparator cache-schema-2 identity. It identifies path/stat metadata,
    # not a content digest. Preserve the directory containing symlinked JSONL.
    files = {source_file.name: source_file}
    video = directory / "camera_detections.mp4"
    if source_file.name == "frames.jsonl" and video.is_file():
        files[video.name] = video
    documents = {}
    for name in ("run_manifest.json", "summary.json"):
        candidate = directory / name
        if candidate.is_file():
            documents[name] = _read_json(candidate)
            files[name] = candidate
    signatures = {name: _signature(file) for name, file in sorted(files.items())}
    fingerprint = None
    if source_file.name == "frames.jsonl" and video.name in files:
        payload = {"cache_schema_version": 2, "source_path": str(directory), "files": signatures}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
    return fingerprint, signatures, documents


def _validate_source_documents(documents: dict) -> None:
    manifest, summary = documents.get("run_manifest.json", {}), documents.get("summary.json", {})
    if manifest.get("status", "complete") != "complete":
        raise InputError("SAM3D run_manifest status must be complete")
    candidates = [summary, manifest.get("summary", {}), manifest.get("config", {})]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise InputError("SAM3D manifest config/summary must be objects")
        for key in ("video_fps", "best_effort_fps"):
            if candidate.get(key) is not None and abs(_number(candidate[key], key) - _FPS) > 0.1:
                raise InputError(f"SAM3D {key} does not support the canonical 20 Hz frame mapping")
    if summary.get("output_frame_count") is not None:
        if _index(summary["output_frame_count"], "output_frame_count") == 0:
            raise InputError("SAM3D output_frame_count must be positive")


def load_sam3_predictions(path: str | Path, required_indices: Iterable[int],
                         expected_fingerprint: str | None = None, *,
                         prediction_policy: str = "annotator-observed") -> tuple[dict[int, dict], dict]:
    """Read requested original/cleaned SAM3D frames, or normalized prediction JSON.

    annotator-observed follows the existing annotation source: observed tracks
    plus finalized retroactive births. all-emitted also retains stale and other
    unobserved tracked states. Neither policy consults GT or extra labels.
    Invalid emitted coordinates/IDs raise; they are never silently discarded.
    JSONL indices must be contiguous from zero through the requested prefix.
    """
    if prediction_policy not in _POLICIES:
        raise InputError(f"prediction_policy must be one of {sorted(_POLICIES)}")
    required = {_index(index, "required frame index") for index in required_indices}
    path = Path(path).expanduser().absolute()
    if path.is_dir():
        candidates = [path / name for name in ("frames.jsonl", "tracks.jsonl", "frames.cleaned.jsonl")]
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    before = _signature(path)
    if path.suffix.lower() == ".json":
        document = _read_json(path)
        frames = _normalized_frames(document, "pred")
        metadata = _document_metadata(document, path, "normalized")
        if expected_fingerprint is not None and metadata["source_fingerprint"] != expected_fingerprint:
            raise InputError("Normalized prediction source_fingerprint does not match GT")
        missing = required - frames.keys()
        if missing:
            raise InputError(f"Missing requested prediction frames: {sorted(missing)}")
        if _signature(path) != before:
            raise InputError("Prediction file changed during reading")
        metadata.update(frame_count=len(required), file_signature=before, prediction_policy="normalized-as-supplied")
        metadata["warnings"].append("Normalized prediction provenance is caller-supplied; no SAM3D source fingerprint is computed.")
        return {index: frames[index] for index in sorted(required)}, metadata

    directory = path.parent.resolve()
    path = directory / path.name
    fingerprint, signatures, documents = _source_identity(directory, path)
    _validate_source_documents(documents)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise InputError(f"SAM3D source fingerprint mismatch: expected {expected_fingerprint}, got {fingerprint}")
    counts = dict.fromkeys(("tracked_records", "fused_records", "emitted_tracked", "emitted_retroactive_fused",
                           "excluded_unobserved_tracked", "excluded_unobserved_stale_tracked",
                           "excluded_nonretroactive_fused", "excluded_unidentified_retroactive_fused",
                           "excluded_fused_already_emitted"), 0)
    frames, previous_stamp, records_read, bytes_read = {}, None, 0, 0
    maximum = max(required, default=-1)
    summary_count = documents.get("summary.json", {}).get("output_frame_count")
    if summary_count is not None and maximum >= summary_count:
        raise InputError(f"Requested frame {maximum} exceeds declared source frame count {summary_count}")
    try:
        with path.open("rb") as stream:
            if maximum >= 0:
                for line_number, line in enumerate(stream, 1):
                    bytes_read += len(line)
                    try:
                        raw = _object(json.loads(line), f"{path} line {line_number}")
                    except (json.JSONDecodeError, UnicodeError) as error:
                        raise InputError(f"Invalid JSON at {path} line {line_number}: {error}") from error
                    index = _index(raw.get("frame_index"), f"line {line_number}.frame_index")
                    if index != line_number - 1:
                        raise InputError(f"Source frame indices must be contiguous: line {line_number} has {index}, expected {line_number - 1}")
                    records_read += 1
                    stamp = _index(raw.get("timestamp_ns"), f"frame {index}.timestamp_ns")
                    if previous_stamp is not None and stamp <= previous_stamp:
                        raise InputError(f"Source timestamps are not strictly increasing at frame {index}")
                    previous_stamp = stamp
                    if index > maximum:
                        break
                    if index in required:
                        frames[index] = _sam3_frame(raw, prediction_policy, counts)
    except OSError as error:
        raise InputError(f"Cannot stream {path}: {error}") from error
    missing = required - frames.keys()
    if missing:
        raise InputError(f"Missing requested source frames: {sorted(missing)}")
    after_fingerprint, after_signatures, _ = _source_identity(directory, path)
    if after_signatures != signatures or after_fingerprint != fingerprint:
        raise InputError("SAM3D source files changed while reading")
    warnings = []
    if fingerprint is None:
        warnings.append("No comparator source fingerprint is available for this JSONL; path/stat provenance is recorded only.")
    if path.name != "frames.jsonl":
        warnings.append("Explicit alternate/cleaned JSONL selected: upstream cropping and yaw repairs may already be present.")
    metadata = {
        "format": "sam3d-jsonl", "path": str(path), "resolved_data_path": str(path.resolve()),
        "source_directory": str(directory), "source_fingerprint": fingerprint,
        "fingerprint_verified_against_gt": expected_fingerprint is not None,
        "fingerprint_method": "comparator cache schema 2: canonical source path and size/mtime_ns of JSONL, video, optional manifest/summary",
        "file_signatures": signatures, "fps": _FPS, "frame_count": len(frames),
        "declared_source_frame_count": summary_count,
        "timestamp_basis": "canonical frame_index / 20; source_timestamp_ns is unchanged capture time",
        "coordinate_system": dict(_COORDINATES), "orientation_convention": dict(_ORIENTATION),
        "prediction_policy": prediction_policy, "selection_counts": counts,
        "selection_counts_scope": "requested frames only; stale exclusion count is a subset of unobserved exclusion count",
        "records_read": records_read, "bytes_read": bytes_read,
        "validation_scope": "contiguous prefix through greatest requested index plus one boundary record, if available; selected people only",
        "warnings": warnings,
    }
    return frames, metadata
