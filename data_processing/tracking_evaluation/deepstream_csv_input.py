"""Strict read-only DeepStream CSV/ledger adapter with exact source-clock joins.

The frame ledger is the authoritative list of processed source ticks. Track
absence on an existing ledger tick means processed-empty; absence of a ledger
tick is unavailable and requires an explicit empty-frame evaluation policy.
No nearest-time matching, coordinate transform, ID repair, ROI crop, heading
flip, confidence cutoff, age cutoff, interpolation, or implicit hold occurs.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from .inputs import InputError


class DeepstreamCSVInputError(InputError):
    """The CSV, frame ledger, or explicit alignment declaration is invalid."""


COORDINATE_SYSTEM = {"plane": "BEV", "reference": "gdc_atrium", "units": "metres"}
CLOCK_ALIGNMENT = {"tracks_stamp": "frame_ledger.stamp_ns", "ledger_stamp": "cam0_record_ns", "gt_source_stamp": "cam0_header_ns"}
MAPPED_CLOCK_ALIGNMENT = {"tracks_stamp": "frame_ledger.stamp_ns", "ledger_stamp": "cam0_record_ns",
    "gt_source_stamp": "source_clock_map.canonical_source_timestamp_ns", "mapped_source_stamp": "source_clock_map.cam0_header_ns"}
TRACK_FIELDS = ["stamp_ns", "track_id", "x", "y", "yaw_rad", "orient_valid", "score"]
LEDGER_FIELDS = ["frame_idx", "stamp_ns"] + [f"cam{cam}_{field}" for cam in range(6) for field in ("header_ns", "record_ns", "repeated")]


def _signature(path):
    try:
        stat = path.stat()
        if not path.is_file():
            raise DeepstreamCSVInputError(f"Required input is not a file: {path}")
        return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
    except OSError as error:
        raise DeepstreamCSVInputError(f"Cannot stat input {path}: {error}") from error


def _read_csv(path, fields):
    before = _signature(path)
    try:
        raw = path.read_bytes()
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True))
    except (OSError, UnicodeError, csv.Error) as error:
        raise DeepstreamCSVInputError(f"Cannot read CSV {path}: {error}") from error
    if _signature(path) != before:
        raise DeepstreamCSVInputError(f"Input changed while reading: {path}")
    if not rows or rows[0] != fields:
        raise DeepstreamCSVInputError(f"Unexpected CSV header in {path}; expected {fields}")
    result = []
    for line, row in enumerate(rows[1:], 2):
        if len(row) != len(fields):
            raise DeepstreamCSVInputError(f"{path}:{line}: expected {len(fields)} fields, got {len(row)}")
        result.append((line, dict(zip(fields, row))))
    return result, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "signature": before}


def _integer(value, context):
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and value and value.isascii() and value.isdecimal():
        return int(value)
    raise DeepstreamCSVInputError(f"{context} must be an exact nonnegative integer; floating timestamps are forbidden")


def _number(value, context, *, nullable=False):
    if nullable and value == "":
        return None
    try:
        if value == "" or value.strip() != value:
            raise ValueError
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DeepstreamCSVInputError(f"{context} must be finite numeric" + (" or empty" if nullable else "")) from error
    if not math.isfinite(result):
        raise DeepstreamCSVInputError(f"{context} must be finite numeric")
    return result


def _read_json_provenance(path):
    path = Path(path).expanduser().resolve()
    before = _signature(path)
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeError, ValueError) as error:
        raise DeepstreamCSVInputError(f"Cannot read JSON {path}: {error}") from error
    if _signature(path) != before:
        raise DeepstreamCSVInputError(f"Input changed while reading: {path}")
    return document, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "signature": before}


def _load_source_clock_map(value, gt_frames):
    """Validate an explicit per-frame map against its actual hashed SAM3 cache."""
    map_base = Path.cwd()
    if isinstance(value, (str, Path)):
        document, map_file = _read_json_provenance(value)
        map_base = Path(map_file["path"]).parent
    elif isinstance(value, Mapping):
        document, map_file = value, None
    else:
        raise DeepstreamCSVInputError("source_clock_map must be a mapping or JSON file path")
    if not isinstance(document, Mapping) or document.get("schema_version") != 1 or isinstance(document.get("schema_version"), bool):
        raise DeepstreamCSVInputError("source_clock_map requires schema_version=1")
    provenance = document.get("source_cache")
    if not isinstance(provenance, Mapping) or not isinstance(provenance.get("path"), str) or not provenance["path"]:
        raise DeepstreamCSVInputError("source_clock_map requires source_cache path and SHA256 provenance")
    cache_path = Path(provenance["path"]).expanduser()
    if not cache_path.is_absolute():
        cache_path = map_base / cache_path
    cache, cache_file = _read_json_provenance(cache_path)
    if provenance.get("sha256") != cache_file["sha256"]:
        raise DeepstreamCSVInputError("source_clock_map cache SHA256 does not match actual source cache")
    if (not isinstance(cache, Mapping) or cache.get("cache_schema_version") != 2
            or not isinstance(cache.get("frames"), list)
            or not isinstance(document.get("source_fingerprint"), str)
            or not document["source_fingerprint"]
            or document["source_fingerprint"] != cache.get("fingerprint")):
        raise DeepstreamCSVInputError("source_clock_map requires matching schema2 cache and source_fingerprint")
    cache_frames = {}
    for frame in cache["frames"]:
        if not isinstance(frame, Mapping):
            raise DeepstreamCSVInputError("Invalid source cache frame")
        index = _integer(frame.get("frame_index"), "source cache frame_index")
        if index in cache_frames:
            raise DeepstreamCSVInputError("Duplicate source cache frame_index")
        cache_frames[index] = frame
    rows = document.get("frames")
    if not isinstance(rows, list) or not rows:
        raise DeepstreamCSVInputError("source_clock_map requires a nonempty frames array")
    mapping, evidence, previous = {}, [], None
    changed = 0
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise DeepstreamCSVInputError("Invalid source_clock_map frame")
        index = _integer(raw.get("frame_index"), "source_clock_map frame_index")
        canonical = _integer(raw.get("canonical_source_timestamp_ns"), "source_clock_map canonical_source_timestamp_ns")
        cam0 = _integer(raw.get("cam0_header_ns"), "source_clock_map cam0_header_ns")
        if index in mapping or (previous is not None and any(a <= b for a, b in zip((index, canonical, cam0), previous))):
            raise DeepstreamCSVInputError("source_clock_map indices and both source clocks must be unique and strictly increasing")
        cached = cache_frames.get(index)
        if not cached or not isinstance(cached.get("source_timestamp_ns_by_camera"), Mapping):
            raise DeepstreamCSVInputError(f"source_clock_map frame {index} lacks source cache per-camera evidence")
        if (canonical != _integer(cached.get("source_timestamp_ns"), "cache canonical source stamp")
                or cam0 != _integer(cached["source_timestamp_ns_by_camera"].get("0"), "cache cam0 source stamp")):
            raise DeepstreamCSVInputError(f"source_clock_map frame {index} disagrees with exact source cache timestamps")
        if index in gt_frames and canonical != _integer(gt_frames[index].get("source_timestamp_ns"), "GT canonical source stamp"):
            raise DeepstreamCSVInputError(f"source_clock_map frame {index} canonical timestamp does not equal GT")
        mapping[index] = cam0
        evidence.append({"frame_index": index, "canonical_source_timestamp_ns": canonical, "cam0_header_ns": cam0})
        changed += index in gt_frames and canonical != cam0
        previous = index, canonical, cam0
    if gt_frames.keys() - mapping.keys():
        raise DeepstreamCSVInputError(f"source_clock_map lacks requested GT indices {sorted(gt_frames.keys() - mapping.keys())}")
    metadata = {"kind": "exact canonical-to-cam0 source cache mapping", "map_file": map_file,
        "source_cache": cache_file, "source_fingerprint": document["source_fingerprint"],
        "frames": evidence, "requested_frames": len(gt_frames), "mapped_source_differs_from_canonical": changed,
        "policy": "Exact integer cache evidence only; no nearest matching, tolerance, timestamp fitting or interpolation. Output retains canonical GT timestamps."}
    return mapping, metadata


def load_deepstream_csv_predictions(tracks_path, frame_ledger_path, gt_frames, *,
                                   absence_policy="error", coordinate_system=None,
                                   clock_alignment=None, calibration_provenance=None,
                                   source_clock_map=None):
    """Return ``({GT_frame_index: normalized_prediction_frame}, metadata)``.

    ``gt_frames`` is the mapping returned by ``load_ground_truth``/selection.
    Coordinates and clocks must be explicitly declared with the exported
    ``COORDINATE_SYSTEM`` and ``CLOCK_ALIGNMENT`` contracts. When canonical GT
    timestamps belong to another camera in a synchronized group, explicitly
    supply ``source_clock_map`` and ``MAPPED_CLOCK_ALIGNMENT``. The mapping is
    verified against its hashed cache's canonical and cam0 per-image stamps;
    output retains the GT timestamp. Calibration
    provenance is a caller-supplied mapping with ``status`` (producer-attested
    or independently-verified) and a nonempty ``description``. This declaration
    does not itself perform calibration verification.

    Both CSVs are completely validated, including rows outside requested GT.
    A non-cam0 view may omit both image timestamps only when explicitly marked
    as repeating a preceding image. Those timestamps remain unknown (None);
    they are never reconstructed from a prior row. Cam0 remains mandatory.
    Exact cam0 image-header stamps identify GT ticks; the tracks' stamp joins
    the ledger's cam0 bag-record stamp. Missing ticks fail by default. Explicit
    ``absence_policy='empty'`` retains missing GT ticks as unavailable empty
    frames without pretending the producer processed them.
    """
    if absence_policy not in {"error", "empty"}:
        raise DeepstreamCSVInputError("absence_policy must be error or empty")
    if coordinate_system != COORDINATE_SYSTEM:
        raise DeepstreamCSVInputError(f"Explicit common-world coordinate_system must equal {COORDINATE_SYSTEM}")
    expected_clock = MAPPED_CLOCK_ALIGNMENT if source_clock_map is not None else CLOCK_ALIGNMENT
    if clock_alignment != expected_clock:
        raise DeepstreamCSVInputError(f"Explicit clock_alignment must equal {expected_clock}")
    if not isinstance(calibration_provenance, Mapping) or calibration_provenance.get("status") not in {"producer-attested", "independently-verified"} or not isinstance(calibration_provenance.get("description"), str) or not calibration_provenance["description"].strip():
        raise DeepstreamCSVInputError("calibration_provenance requires status producer-attested/independently-verified and a nonempty description")
    if not isinstance(gt_frames, Mapping) or not gt_frames:
        raise DeepstreamCSVInputError("gt_frames must be a nonempty mapping of normalized GT frames")
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in gt_frames):
        raise DeepstreamCSVInputError("GT mapping keys must be nonnegative integers")
    mapped_clocks, map_meta = (None, None) if source_clock_map is None else _load_source_clock_map(source_clock_map, gt_frames)
    tracks_path, frame_ledger_path = (Path(p).expanduser().resolve() for p in (tracks_path, frame_ledger_path))
    ledger_rows, ledger_meta = _read_csv(frame_ledger_path, LEDGER_FIELDS)
    track_rows, tracks_meta = _read_csv(tracks_path, TRACK_FIELDS)
    if not ledger_rows:
        raise DeepstreamCSVInputError("Frame ledger contains no processed frames")
    by_stamp, by_header = {}, {}
    previous, repeated_counts = None, Counter()
    last_known_camera_clocks = {}
    unknown_repeat_counts, unknown_repeat_events = Counter(), []
    for line, raw in ledger_rows:
        context = f"{frame_ledger_path}:{line}"
        row = {key: _integer(raw[key], context + "." + key) for key in ("frame_idx", "stamp_ns")}
        for cam in range(6):
            flag_key = f"cam{cam}_repeated"
            flag = _integer(raw[flag_key], context + "." + flag_key)
            if flag not in (0, 1):
                raise DeepstreamCSVInputError(f"{context}: repeated flag must be 0 or 1")
            row[flag_key] = flag
            names = [f"cam{cam}_{kind}" for kind in ("header_ns", "record_ns")]
            blanks = [raw[name] == "" for name in names]
            if any(blanks):
                if cam == 0 or not all(blanks) or flag != 1 or previous is None:
                    raise DeepstreamCSVInputError(f"{context}: paired missing camera timestamps require non-cam0 repeated=1 after a preceding ledger frame")
                row.update({name: None for name in names})
                unknown_repeat_counts[str(cam)] += 1
                unknown_repeat_events.append({"producer_frame_index": row["frame_idx"], "camera": cam,
                    "header_ns": None, "record_ns": None, "source_csv_line": line})
            else:
                row.update({name: _integer(raw[name], context + "." + name) for name in names})
        if row["stamp_ns"] != row["cam0_record_ns"]:
            raise DeepstreamCSVInputError(f"{context}: ledger stamp_ns differs from declared cam0_record_ns")
        if row["stamp_ns"] in by_stamp or row["cam0_header_ns"] in by_header:
            raise DeepstreamCSVInputError(f"{context}: duplicate ledger stamp or cam0 source timestamp")
        if previous is not None and (row["frame_idx"] <= previous["frame_idx"] or row["stamp_ns"] <= previous["stamp_ns"] or row["cam0_header_ns"] <= previous["cam0_header_ns"]):
            raise DeepstreamCSVInputError(f"{context}: ledger frame indices and cam0 clocks must strictly increase")
        for cam in range(6):
            flag = row[f"cam{cam}_repeated"]
            clocks = tuple(row[f"cam{cam}_{kind}"] for kind in ("header_ns", "record_ns"))
            known = clocks[0] is not None
            prior_known = last_known_camera_clocks.get(cam)
            if flag and known:
                if previous is None or prior_known is None or clocks != prior_known:
                    raise DeepstreamCSVInputError(f"{context}: repeated camera flag does not refer to its previous exact image")
            if known and prior_known is not None and any(current < old for current, old in zip(clocks, prior_known)):
                raise DeepstreamCSVInputError(f"{context}: camera image clocks go backward")
            if known:
                last_known_camera_clocks[cam] = clocks
            repeated_counts[str(cam)] += flag
        by_stamp[row["stamp_ns"]] = row
        by_header[row["cam0_header_ns"]] = row
        previous = row
    observations = {stamp: [] for stamp in by_stamp}
    seen, total_known_yaw, total_unknown_score = set(), 0, 0
    previous_track_stamp = None
    for line, raw in track_rows:
        context = f"{tracks_path}:{line}"
        stamp = _integer(raw["stamp_ns"], context + ".stamp_ns")
        _integer(raw["track_id"], context + ".track_id")
        tid = raw["track_id"]  # Preserve identity text; never alias it with GT IDs.
        if stamp not in by_stamp:
            raise DeepstreamCSVInputError(f"{context}: track timestamp has no processed ledger frame")
        if previous_track_stamp is not None and stamp < previous_track_stamp:
            raise DeepstreamCSVInputError(f"{context}: track timestamps must not decrease")
        previous_track_stamp = stamp
        if (stamp, tid) in seen:
            raise DeepstreamCSVInputError(f"{context}: duplicate timestamp/track_id")
        seen.add((stamp, tid))
        if raw["orient_valid"] not in ("0", "1"):
            raise DeepstreamCSVInputError(f"{context}: orient_valid must be 0 or 1")
        raw_yaw = _number(raw["yaw_rad"], context + ".yaw_rad", nullable=True)
        if raw["orient_valid"] == "1" and raw_yaw is None:
            raise DeepstreamCSVInputError(f"{context}: valid orientation requires a finite yaw")
        yaw = raw_yaw if raw["orient_valid"] == "1" else None
        score = _number(raw["score"], context + ".score", nullable=True)
        if score is not None and not 0 <= score <= 1:
            raise DeepstreamCSVInputError(f"{context}: score must be in [0,1] or empty")
        observations[stamp].append({"track_id": tid, "x": _number(raw["x"], context + ".x"),
            "y": _number(raw["y"], context + ".y"), "yaw": yaw, "score": score,
            "orient_valid": raw["orient_valid"] == "1", "source_yaw_rad": raw_yaw})
        total_known_yaw += yaw is not None
        total_unknown_score += score is None
    output, unavailable, processed_empty, gt_stamps = {}, [], [], set()
    selected_known_yaw, selected_observations = 0, 0
    previous_gt_stamp = None
    for index, gt in sorted(gt_frames.items()):
        if not isinstance(gt, Mapping) or not isinstance(gt.get("frame_index"), int) or isinstance(gt.get("frame_index"), bool) or gt.get("frame_index") != index:
            raise DeepstreamCSVInputError("GT mapping keys must match nonnegative integer frame_index fields")
        stamp = _integer(gt.get("source_timestamp_ns"), f"GT frame {index}.source_timestamp_ns")
        if stamp in gt_stamps:
            raise DeepstreamCSVInputError("GT source timestamps must be unique")
        if previous_gt_stamp is not None and stamp <= previous_gt_stamp:
            raise DeepstreamCSVInputError("GT source timestamps must increase with frame index")
        gt_stamps.add(stamp)
        previous_gt_stamp = stamp
        timestamp = gt.get("timestamp_s")
        if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp)):
            raise DeepstreamCSVInputError(f"GT frame {index}.timestamp_s must be finite or null")
        mapped_stamp = stamp if mapped_clocks is None else mapped_clocks[index]
        ledger = by_header.get(mapped_stamp)
        base = {"frame_index": index, "timestamp_s": timestamp, "source_timestamp_ns": stamp}
        if mapped_clocks is not None:
            base["mapped_cam0_source_timestamp_ns"] = mapped_stamp
        if ledger is None:
            unavailable.append(index)
            output[index] = {**base, "pred": [], "availability": "unavailable", "processed": False,
                "unavailable_reason": "No exact cam0 source timestamp in producer frame ledger"}
            continue
        people = observations[ledger["stamp_ns"]]
        if not people:
            processed_empty.append(index)
        output[index] = {**base, "pred": people, "availability": "processed" if people else "processed-empty", "processed": True,
            "producer_frame_index": ledger["frame_idx"], "producer_stamp_ns": ledger["stamp_ns"],
            "camera_source_ledger": dict(ledger)}
        selected_observations += len(people)
        selected_known_yaw += sum(p["yaw"] is not None for p in people)
    if unavailable and absence_policy == "error":
        raise DeepstreamCSVInputError(f"No exact producer frame for GT indices {unavailable}; explicitly set absence_policy='empty' to score unavailable frames as empty")
    for path, metadata in ((tracks_path, tracks_meta), (frame_ledger_path, ledger_meta)):
        if _signature(path) != metadata["signature"]:
            raise DeepstreamCSVInputError(f"Input changed during complete adapter read: {path}")
    if map_meta is not None:
        for record in (map_meta["map_file"], map_meta["source_cache"]):
            if record is not None and _signature(Path(record["path"])) != record["signature"]:
                raise DeepstreamCSVInputError(f"Source clock map input changed during complete adapter read: {record['path']}")
    metadata = {"format": "deepstream-csv-with-frame-ledger", "path": str(tracks_path), "frame_ledger_path": str(frame_ledger_path),
        "files": {"tracks": tracks_meta, "frame_ledger": ledger_meta}, "coordinate_system": dict(coordinate_system),
        "clock_alignment": dict(clock_alignment), "calibration_provenance": dict(calibration_provenance),
        "orientation_convention": {"units": "radians", "zero": "+X", "positive": "counterclockwise toward +Y"},
        "yaw_policy": "Preserve producer yaw exactly when orient_valid=1; unknown otherwise. No flip or velocity-derived yaw.",
        "prediction_policy": "all-emitted; no confidence/age filtering, interpolation, hold, ID repair, or ROI crop",
        "repeated_camera_timestamp_policy": "Paired blank non-cam0 timestamps are allowed only with repeated=1 after a preceding ledger frame; preserved as null, never inferred. Known camera clocks remain monotonic across unknown entries.",
        "repeated_camera_unknown_timestamp_events": unknown_repeat_events,
        "validation_scope": "Complete tracks and frame-ledger files, including records outside requested GT frames",
        "absence_policy": absence_policy, "frame_count": len(output), "matched_producer_frames": len(output)-len(unavailable),
        "unavailable_frame_indices": unavailable, "processed_empty_frame_indices": processed_empty,
        "selected_prediction_observations": selected_observations, "selected_known_yaw_observations": selected_known_yaw,
        "full_file_counts": {"ledger_frames": len(by_stamp), "track_rows": len(track_rows),
            "processed_empty_frames": sum(not p for p in observations.values()), "known_yaw": total_known_yaw,
            "unknown_yaw": len(track_rows)-total_known_yaw, "unknown_scores": total_unknown_score,
            "camera_repeated_flags": dict(repeated_counts),
            "camera_unknown_repeated_timestamp_pairs": {str(cam): unknown_repeat_counts[str(cam)] for cam in range(6)}},
        "warnings": ["Coordinate, clock, and calibration declarations are caller supplied; the adapter verifies file structure and exact clock joins, not physical calibration."]}
    if unavailable:
        metadata["warnings"].append(f"{len(unavailable)} GT samples are unavailable in the producer ledger and are explicitly retained as empty predictions.")
    if unknown_repeat_events:
        metadata["warnings"].append(f"{len(unknown_repeat_events)} explicitly repeated non-reference camera images omit both timestamps; kept unknown. Cam0 alignment remains exact.")
    if map_meta is not None:
        metadata["source_clock_map"] = map_meta
    return output, metadata
