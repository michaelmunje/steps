"""Published, pose-preserving SAM3 cleanup with explicit original provenance.

The published receipt binds the derived file to the original by file metadata.
The selected original records are also checked, so a changed timestamp, ID or
position cannot silently turn cleanup into a different prediction sequence.
No source, annotations, or cleanup products are written by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .inputs import (InputError, _index, _number, _object, _read_json, _signature,
                     load_sam3_predictions)


def _publication_stat(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino, "device": stat.st_dev}


def _bounds(values, context: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise InputError(f"{context} must be [xmin, xmax, ymin, ymax]")
    result = [_number(value, context) for value in values]
    if result[0] > result[1] or result[2] > result[3]:
        raise InputError(f"{context} minima must not exceed maxima")
    return result


def load_cleaned_sam3_predictions(
    path: str | Path, required_indices: Iterable[int], expected_fingerprint: str,
    *, original_source: str | Path, evaluation_roi: list[float],
    prediction_policy: str = "annotator-observed",
    cleanup_report: str | Path | None = None,
    cleanup_receipt: str | Path | None = None,
) -> tuple[dict[int, dict], dict]:
    """Validate published cleanup, load requested records and retain clean yaw.

    ``original_source`` is the annotation comparator's source directory, whose
    path participates in its fingerprint (do not replace it with the JSONL's
    resolved target directory). Sidecars default to the published JSONL basename
    with ``.report.json`` / ``.receipt.json`` suffixes. Evaluation must explicitly
    stay inside the cleanup crop. Receipt checks use the publisher's stat-based
    provenance, not a new whole-file content hash of a multi-gigabyte recording.
    """
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise InputError("Cleaned SAM3 requires GT's original source_fingerprint")
    required = {_index(index, "required frame index") for index in required_indices}
    path = Path(path).expanduser().absolute()
    if path.suffix.lower() != ".jsonl":
        raise InputError("Cleaned SAM3 predictions must be a published JSONL file")
    original = Path(original_source).expanduser().absolute()
    original_file = original / "frames.jsonl" if original.is_dir() else original
    if original_file.name != "frames.jsonl":
        raise InputError("Original SAM3 source must be its comparator directory or frames.jsonl")
    report_path = Path(cleanup_report).expanduser().absolute() if cleanup_report else path.with_suffix(".report.json")
    receipt_path = Path(cleanup_receipt).expanduser().absolute() if cleanup_receipt else path.with_suffix(".receipt.json")
    watched = [path, original_file, report_path, receipt_path]
    before = {str(file): _signature(file) for file in watched}
    report, receipt = _read_json(report_path), _read_json(receipt_path)
    if receipt.get("publication_version") != 1 or isinstance(receipt.get("publication_version"), bool):
        raise InputError("Unsupported cleaned SAM3 publication_version")
    if report.get("format") != "pose_preserving_tracks_v1" or report.get("input_stat_unchanged") is not True:
        raise InputError("Cleanup must declare completed pose_preserving_tracks_v1 with unchanged input")
    if receipt.get("source") != str(original_file.resolve()) or report.get("input") != str(original_file.resolve()):
        raise InputError("Cleanup source path does not match the explicit original SAM3 source")
    if report.get("output") != str(path.resolve()):
        raise InputError("Cleanup report output path does not match the selected JSONL")
    if receipt.get("source_stat") != _publication_stat(original_file):
        raise InputError("Original SAM3 source no longer matches the cleanup receipt source_stat")
    if receipt.get("published_stat") != _publication_stat(path):
        raise InputError("Cleaned SAM3 file no longer matches the publication receipt published_stat")
    if (report.get("input_bytes") != before[str(original_file)]["size"]
            or report.get("output_bytes") != before[str(path)]["size"]):
        raise InputError("Cleanup report byte counts do not match the source and derived files")

    publication = _object(report.get("publication"), "cleanup report.publication")
    for key in ("method", "staging_output", "staging_report_sha256", "copied_sha256"):
        if key not in publication or publication[key] != receipt.get(key):
            raise InputError(f"Cleanup report and receipt disagree on publication.{key}")
    if publication["method"] not in {"hardlink", "copy"}:
        raise InputError("Unsupported cleaned SAM3 publication method")
    # Publisher writes the public report as the staging report with just these
    # two fields changed. Reconstruct its exact canonical writer serialization;
    # this checks the receipt's report digest without requiring staging files.
    staging_report = {key: value for key, value in report.items() if key != "publication"}
    staging_report["output"] = publication["staging_output"]
    report_digest = hashlib.sha256((json.dumps(staging_report, indent=2) + "\n").encode()).hexdigest()
    if report_digest != publication["staging_report_sha256"]:
        raise InputError("Cleanup report does not match the publication receipt's staging report digest")

    config = _object(report.get("config"), "cleanup config")
    cleanup_roi = _bounds([config.get(key) for key in ("x_min", "x_max", "y_min", "y_max")], "cleanup ROI")
    roi = _bounds(evaluation_roi, "Explicit evaluation ROI for cropped predictions")
    if not (cleanup_roi[0] <= roi[0] <= roi[1] <= cleanup_roi[1]
            and cleanup_roi[2] <= roi[2] <= roi[3] <= cleanup_roi[3]):
        raise InputError("Evaluation ROI extends outside the cleaned SAM3 crop")
    declared_count = _index(report.get("frames"), "cleanup frames")
    if max(required, default=-1) >= declared_count:
        raise InputError("Requested frame exceeds the cleanup report's frame count")

    original_frames, original_meta = load_sam3_predictions(
        original, required, expected_fingerprint=expected_fingerprint,
        prediction_policy=prediction_policy)
    if original_meta.get("declared_source_frame_count") not in (None, declared_count):
        raise InputError("Cleanup and original source declared frame counts differ")
    cleaned_frames, cleaned_meta = load_sam3_predictions(path, required, prediction_policy=prediction_policy)
    people_checked = 0
    for index, frame in cleaned_frames.items():
        source = original_frames[index]
        if frame["source_timestamp_ns"] != source["source_timestamp_ns"]:
            raise InputError(f"Cleaned SAM3 source timestamp mismatch at frame {index}")
        original_people = {person["track_id"]: person for person in source["pred"]}
        for person in frame["pred"]:
            ident = person["track_id"]
            original_person = original_people.get(ident)
            if original_person is None:
                raise InputError(f"Cleaned SAM3 introduces prediction ID {ident!r} at frame {index}")
            if any(person[axis] != original_person[axis] for axis in ("x", "y")):
                raise InputError(f"Cleaned SAM3 changes position for ID {ident!r} at frame {index}")
            if not (cleanup_roi[0] <= person["x"] <= cleanup_roi[1]
                    and cleanup_roi[2] <= person["y"] <= cleanup_roi[3]):
                raise InputError(f"Cleaned SAM3 position outside declared crop at frame {index}, ID {ident!r}")
            people_checked += 1
    if any(_signature(file) != before[str(file)] for file in watched):
        raise InputError("Cleaned SAM3 provenance files changed while reading")
    # source_fingerprint remains the verified ORIGINAL identity; a separate
    # derived fingerprint makes the cleanup variant visible in every report.
    provenance = {"original_source_fingerprint": original_meta["source_fingerprint"],
                  "derived_path": str(path), "derived_signature": before[str(path)],
                  "cleanup_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                  "cleanup_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
    if any(_signature(file) != before[str(file)] for file in watched):
        raise InputError("Cleaned SAM3 provenance files changed while recording digests")
    metadata = {**cleaned_meta, "format": "sam3d-cleaned-jsonl",
                "source_fingerprint": original_meta["source_fingerprint"],
                "fingerprint_verified_against_gt": True,
                "fingerprint_role": "original source identity; derived product is separately identified",
                "fingerprint_method": original_meta["fingerprint_method"],
                "derived_fingerprint": hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
                "derived_provenance": provenance, "cleanup_config": config,
                "cleanup_roi_m": cleanup_roi, "evaluation_roi_m": roi,
                "cleanup_report": str(report_path), "cleanup_receipt": str(receipt_path),
                "original_input": original_meta,
                "original_compatibility_checks": {"frames": len(cleaned_frames), "people": people_checked,
                    "checks": ["exact frame indices and capture timestamps", "unchanged prediction IDs and XY",
                               "emitted coordinates inside declared cleanup crop"]},
                "heading_policy": "use cleaned yaw_ground_rad verbatim; never restore original yaw or use GT headings",
                "warnings": ["Derived SAM3 predictions include upstream cropping and yaw repairs; compare as a separately named variant.",
                             "Publication provenance uses file path/stat metadata plus the cleanup report digest; whole data-file hashes are not recomputed."]}
    return cleaned_frames, metadata
