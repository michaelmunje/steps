"""Command-line entry point; inputs are read-only and report directories are new."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def select_frames(frames, selection):
    """Select only supplied GT snapshots; absent reviewed frames stay absent."""
    start = selection.get("start_s")
    end = selection.get("end_s")
    if start is not None:
        start = _finite(start, "start_s")
    if end is not None:
        end = _finite(end, "end_s")
    if start is not None and end is not None and end < start:
        raise ValueError("end_s must be at least start_s")
    step, origin = selection.get("frame_step", 1), selection.get("frame_origin", 0)
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("frame_step must be a positive integer")
    if isinstance(origin, bool) or not isinstance(origin, int):
        raise ValueError("frame_origin must be an integer")
    selected = {}
    for index, frame in sorted(frames.items()):
        if (index - origin) % step:
            continue
        timestamp = frame.get("timestamp_s")
        if start is not None or end is not None:
            if timestamp is None:
                raise ValueError("Time-window selection requires a timestamp_s for every candidate frame")
            timestamp = _finite(timestamp, "timestamp_s")
            if start is not None and timestamp < start - 1e-9:
                continue
            if end is not None and timestamp > end + 1e-9:
                continue
        selected[index] = frame
    if not selected:
        raise ValueError("No ground-truth frames remain. Empty reviewed exports are not complete baselines.")
    coverage = {"input_gt_frames": len(frames), "selected_gt_frames": len(selected),
                "selected_frame_indices": list(selected), "selection": dict(selection),
                "missing_gt_policy": "exclude unsupplied GT frames from every metric; never interpret them as empty scenes"}
    fps = selection.get("source_fps")
    if fps is not None:
        fps = _finite(fps, "source_fps")
        if fps <= 0:
            raise ValueError("source_fps must be positive")
        # This explicitly requested grid uses source frame zero as time zero.
        for index, frame in selected.items():
            if frame.get("timestamp_s") is None or abs(frame["timestamp_s"] - index / fps) > 1e-6:
                raise ValueError("source_fps grid requires timestamp_s = frame_index / source_fps")
        if start is not None and end is not None:
            first, last = math.ceil(start * fps - 1e-8), math.floor(end * fps + 1e-8)
            expected = [i for i in range(first, last + 1) if (i - origin) % step == 0]
            coverage["requested_grid_frames"] = len(expected)
            coverage["missing_gt_frame_indices"] = [i for i in expected if i not in selected]
            coverage["selected_grid_fraction"] = len(selected) / len(expected) if expected else None
    return selected, coverage


def apply_roi(sequence, roi):
    if roi is None:
        return sequence, {"scope": "full scene", "gt_removed": 0, "pred_removed": 0}
    if len(roi) != 4:
        raise ValueError("ROI must be [xmin, xmax, ymin, ymax]")
    xmin, xmax, ymin, ymax = [_finite(x, "ROI bound") for x in roi]
    if xmin > xmax or ymin > ymax:
        raise ValueError("ROI minima must not exceed maxima")
    result = copy.deepcopy(sequence)
    removed = {"scope": "inclusive rectangle applied independently to GT and predictions",
               "bounds_m": [xmin, xmax, ymin, ymax], "gt_removed": 0, "pred_removed": 0}
    for frame in result["frames"]:
        for role in ("gt", "pred"):
            keep = [node for node in frame[role]
                    if xmin <= _finite(node["x"], "x") <= xmax
                    and ymin <= _finite(node["y"], "y") <= ymax]
            removed[role + "_removed"] += len(frame[role]) - len(keep)
            frame[role] = keep
    return result, removed


def join_frames(gt_frames, predictions):
    missing = sorted(set(gt_frames) - set(predictions))
    if missing:
        raise ValueError(f"Prediction records missing for GT frames {missing[:20]}; explicit empty prediction frames are required")
    result = []
    for index, gt in sorted(gt_frames.items()):
        pred = predictions[index]
        gt_stamp, pred_stamp = gt.get("source_timestamp_ns"), pred.get("source_timestamp_ns")
        if gt_stamp is not None and pred_stamp is not None and int(gt_stamp) != int(pred_stamp):
            raise ValueError(f"Source timestamp mismatch at frame {index}")
        gt_time, pred_time = gt.get("timestamp_s"), pred.get("timestamp_s")
        if gt_time is not None and pred_time is not None and abs(gt_time - pred_time) > 1e-6:
            raise ValueError(f"Relative timestamp mismatch at frame {index}")
        result.append({"frame_index": index, "timestamp_s": gt.get("timestamp_s"),
                       "gt": gt["gt"], "pred": pred["pred"]})
    return result


def _resolve(path, base):
    value = Path(path).expanduser()
    return str(value if value.is_absolute() else base / value)


def _load_manifest(args):
    if args.manifest:
        path = Path(args.manifest).resolve()
        document = json.loads(path.read_text())
        entries = document.get("sequences")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Manifest must contain a nonempty sequences array")
        base = path.parent
    else:
        if not args.gt or not args.pred:
            raise ValueError("Use --manifest or supply both --gt and --pred")
        document = {"sequences": [{"name": args.name or Path(args.gt).parent.name,
                                  "ground_truth": args.gt, "predictions": args.pred,
                                  "prediction_format": args.prediction_format or "sam3",
                                  "alignment": args.alignment,
                                  "deepstream_frame_ledger": args.deepstream_frame_ledger,
                                  "sam3_original_source": args.sam3_original_source,
                                  "sam3_cleanup_report": args.sam3_cleanup_report,
                                  "sam3_cleanup_receipt": args.sam3_cleanup_receipt,
                                  "prediction_policy": args.sam3_policy or "annotator-observed"}]}
        entries, base = document["sequences"], Path.cwd()
    names = set()
    for original in entries:
        entry = copy.deepcopy(original)
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Every sequence needs a nonempty, unique name")
        names.add(name)
        for field in ("ground_truth", "predictions", "alignment", "normalized_sequence",
                      "sam3_original_source", "sam3_cleanup_report", "sam3_cleanup_receipt",
                      "deepstream_frame_ledger"):
            if entry.get(field):
                entry[field] = _resolve(entry[field], base)
        if args.sam3_policy is not None:
            entry["prediction_policy"] = args.sam3_policy
        if args.prediction_format is not None:
            entry["prediction_format"] = args.prediction_format
        if args.alignment is not None:
            entry["alignment"] = _resolve(args.alignment, Path.cwd())
        for field in ("sam3_original_source", "sam3_cleanup_report", "sam3_cleanup_receipt",
                      "deepstream_frame_ledger"):
            if getattr(args, field) is not None:
                entry[field] = _resolve(getattr(args, field), Path.cwd())
        selection = dict(document.get("selection", {}))
        selection.update(entry.get("selection", {}))
        for key in ("start_s", "end_s", "frame_step", "frame_origin", "source_fps"):
            if getattr(args, key) is not None:
                selection[key] = getattr(args, key)
        entry["selection"] = selection
        entry["roi"] = args.roi if args.roi is not None else entry.get("roi", document.get("roi"))
        yield entry, document


def run_sequence(entry, config):
    from .inputs import load_ground_truth, load_sam3_predictions
    from .metrics import evaluate_sequence
    wall_start = time.perf_counter()
    if entry.get("normalized_sequence"):
        path = Path(entry["normalized_sequence"])
        frames, gt_meta = load_ground_truth(path)
        gt_frames, coverage = select_frames(frames, entry["selection"])
        load_gt_s = time.perf_counter() - wall_start
        pred_start = time.perf_counter()
        predictions, pred_meta = load_sam3_predictions(path, set(gt_frames))
        sequence = {"name": entry["name"], "frames": join_frames(gt_frames, predictions)}
        load_pred_s = time.perf_counter() - pred_start
    else:
        if not entry.get("ground_truth") or not entry.get("predictions"):
            raise ValueError("Sequence requires ground_truth and predictions paths")
        frames, gt_meta = load_ground_truth(entry["ground_truth"])
        gt_frames, coverage = select_frames(frames, entry["selection"])
        load_gt_s = time.perf_counter() - wall_start
        pred_start = time.perf_counter()
        format_name = entry.get("prediction_format", "sam3")
        if format_name == "deepstream":
            from .deepstream_input import load_deepstream_predictions
            if not entry.get("alignment"):
                raise ValueError("DeepStream comparison requires an explicit --alignment manifest")
            alignment = json.loads(Path(entry["alignment"]).read_text())
            dataset_id = gt_meta.get("session", {}).get("dataset_id")
            if dataset_id is not None and alignment.get("sequence_id") != dataset_id:
                raise ValueError("DeepStream alignment sequence_id does not match the GT dataset_id")
            predictions, pred_meta = load_deepstream_predictions(entry["predictions"], gt_frames, alignment)
        elif format_name == "deepstream-csv":
            from .deepstream_csv_input import load_deepstream_csv_predictions
            if not entry.get("alignment") or not entry.get("deepstream_frame_ledger"):
                raise ValueError("DeepStream CSV requires --alignment and --deepstream-frame-ledger")
            alignment_path = Path(entry["alignment"])
            alignment_bytes = alignment_path.read_bytes()
            alignment = json.loads(alignment_bytes)
            dataset_id = gt_meta.get("session", {}).get("dataset_id")
            if dataset_id is not None and alignment.get("sequence_id") != dataset_id:
                raise ValueError("DeepStream CSV alignment sequence_id does not match the GT dataset_id")
            source_clock_map = alignment.get("source_clock_map")
            if isinstance(source_clock_map, str):
                source_clock_map = _resolve(source_clock_map, alignment_path.resolve().parent)
            predictions, pred_meta = load_deepstream_csv_predictions(
                entry["predictions"], entry["deepstream_frame_ledger"], gt_frames,
                absence_policy=alignment.get("absence_policy", "error"),
                coordinate_system=alignment.get("coordinate_system"),
                clock_alignment=alignment.get("clock_alignment"),
                calibration_provenance=alignment.get("calibration_provenance"),
                source_clock_map=source_clock_map)
            import hashlib
            if alignment_path.read_bytes() != alignment_bytes:
                raise ValueError("DeepStream CSV alignment manifest changed while loading")
            pred_meta["alignment_manifest"] = {
                "path": str(alignment_path),
                "sha256": hashlib.sha256(alignment_bytes).hexdigest(),
                "declarations": alignment}
        elif format_name == "sam3-cleaned":
            from .cleaned_sam3_input import load_cleaned_sam3_predictions
            if not entry.get("sam3_original_source"):
                raise ValueError("Cleaned SAM3 requires explicit --sam3-original-source provenance")
            predictions, pred_meta = load_cleaned_sam3_predictions(
                entry["predictions"], set(gt_frames), gt_meta.get("source_fingerprint"),
                original_source=entry["sam3_original_source"], evaluation_roi=entry.get("roi"),
                prediction_policy=entry.get("prediction_policy", "annotator-observed"),
                cleanup_report=entry.get("sam3_cleanup_report"),
                cleanup_receipt=entry.get("sam3_cleanup_receipt"))
        elif format_name in ("sam3", "normalized"):
            fingerprint = gt_meta.get("source_fingerprint", gt_meta.get("session", {}).get("source_fingerprint"))
            predictions, pred_meta = load_sam3_predictions(
                entry["predictions"], set(gt_frames), expected_fingerprint=fingerprint,
                prediction_policy=entry.get("prediction_policy", "annotator-observed"))
        else:
            raise ValueError(f"Unsupported prediction_format: {format_name}")
        sequence = {"name": entry["name"], "frames": join_frames(gt_frames, predictions)}
        load_pred_s = time.perf_counter() - pred_start
    gt_coordinates = gt_meta.get("coordinate_system") or {}
    pred_coordinates = pred_meta.get("coordinate_system") or {}
    gt_reference, pred_reference = gt_coordinates.get("reference"), pred_coordinates.get("reference")
    if gt_reference is not None and pred_reference is not None and gt_reference != pred_reference:
        raise ValueError("GT and prediction inputs declare different world coordinate references")
    sequence, roi_meta = apply_roi(sequence, entry.get("roi"))
    metric_start = time.perf_counter()
    result = evaluate_sequence(sequence, config)
    metric_s = time.perf_counter() - metric_start
    runtime = {"load_gt_s": load_gt_s, "load_predictions_s": load_pred_s,
               "metrics_s": metric_s, "sequence_wall_s": time.perf_counter() - wall_start,
               "metric_frames_per_second": len(sequence["frames"]) / metric_s if metric_s else None}
    return {"name": entry["name"], "metrics": result, "coverage": coverage,
            "roi": roi_meta, "ground_truth_input": gt_meta, "prediction_input": pred_meta,
            "runtime": runtime, "notes": entry.get("notes", [])}


def _display(value, digits=4):
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_report(report):
    lines = [f"# Tracking comparison — {report['label']}", "", f"Created: {report['created_at']}", "",
             "All scores describe the supplied GT snapshots. Missing GT frames are excluded, not empty scenes. "
             "Prediction IDs are preserved; GT and prediction IDs use separate namespaces.", ""]
    for note in report.get("notes", []):
        lines.extend([f"- {note}"])
    lines.extend(["", "## Results", "", "Metrics use ratios from 0 to 1 unless indicated; MOTA may be negative. "
                  "MOTP is Euclidean distance in metres; lower is better. HOTA uses the reported world-distance similarity, not box IoU.", "",
                  "HOTA similarity is `max(0, 1 - distance_m / similarity_scale_m)` for `linear`, or "
                  "`exp(-0.5 * (distance_m / similarity_scale_m)^2)` for `gaussian`. Both are zero when "
                  "`distance_m > max_distance_m`. The selected method, numeric distance gate, scale, and alpha thresholds "
                  "are recorded below under Protocol and runtime.", "",
                  "MOTA and ID switches below use TrackEval CLEAR correspondence preference with the declared "
                  "world-distance gate. Position and heading errors retain independent distance-based matches. "
                  "The historical distance-assignment MOTA/IDSW remain in JSON for compatibility and are not the CLEAR scores.", "",
                  "| Sequence | Frames | HOTA | DetA | AssA | IDF1 | CLEAR MOTA | Position MAE (m) | Yaw MAE (deg) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    rows = [(s["name"], s["metrics"]) for s in report["sequences"]] + [("Combined", report["combined"])]
    for name, metrics in rows:
        values = [metrics.get(k) for k in ("frame_count", "HOTA", "DetA", "AssA", "IDF1")]
        values += [metrics.get("clear", {}).get("MOTA"), metrics.get("MOTP_m"), metrics.get("yaw_MAE_deg")]
        lines.append("| " + name.replace("|", "\\|") + " | " + " | ".join(_display(v) for v in values) + " |")
    lines.extend(["", "## Counts and coverage", ""])
    for sequence in report["sequences"]:
        m, c = sequence["metrics"], sequence["coverage"]
        lines.extend([f"### {sequence['name']}", "",
                      f"GT observations: {m['total_GT_observations']}; predictions: {m['total_prediction_observations']}; "
                      f"Distance-assignment TP/FP/FN: {m['TP']}/{m['FP']}/{m['FN']}.", "",
                      f"IDTP/IDFP/IDFN: {m['IDTP']}/{m['IDFP']}/{m['IDFN']}; "
                      f"precision: {_display(m['precision'])}; recall: {_display(m['recall'])}.", "",
                      f"Heading pairs: {m['yaw_matched_observations']} of {m['matched_observations']} spatial matches "
                      f"(coverage {_display(m['yaw_coverage'])}). Unknown headings remain unknown.", "",
                      f"Selected GT frames: {c['selected_gt_frames']} of {c['input_gt_frames']} supplied; "
                      f"missing requested grid indices: {c.get('missing_gt_frame_indices', 'grid not specified')}.", "",
                      f"Scope: {sequence['roi']}. Runtime: {sequence['runtime']}.", ""])
        if "clear" in m:
            clear = m["clear"]
            lines.extend([f"CLEAR TP/FP/FN: {clear['TP']}/{clear['FP']}/{clear['FN']}; "
                          f"CLEAR ID switches: {clear['IDSW']}; fragmentation: {clear['Frag']}; "
                          f"CLEAR MOTA: {_display(clear['MOTA'])}.", ""])
        for note in sequence.get("notes", []):
            lines.append(f"- {note}")
        for role in ("ground_truth_input", "prediction_input"):
            for warning in sequence[role].get("warnings", []) + sequence[role].get("limitations", []):
                lines.append(f"- {warning}")
        lines.append("")
    lines.extend(["## Protocol and runtime", "", "```json", json.dumps({"metric_config": report["metric_config"],
                  "protocol": report["combined"].get("protocol"),
                  "clear_protocol": report["combined"].get("clear", {}).get("protocol"),
                  "runtime": report["runtime"]}, indent=2), "```", "",
                  "HOTA/IDF1 procedures: [official TrackEval reference](https://github.com/JonathonLuiten/TrackEval). "
                  "The JSON report includes each HOTA threshold, additive sums, provenance, exact frame coverage, and per-sequence results.", ""])
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare", help="Compare reviewed GT against explicitly selected predictions")
    source = compare.add_mutually_exclusive_group(required=True)
    source.add_argument("--gt")
    source.add_argument("--manifest", help="JSON with a sequences array; relative paths resolve beside it")
    compare.add_argument("--pred")
    compare.add_argument("--name")
    compare.add_argument("--prediction-format", choices=("sam3", "sam3-cleaned", "normalized", "deepstream", "deepstream-csv"))
    compare.add_argument("--deepstream-frame-ledger", help="DeepStream CSV frame ledger mapping output stamps to camera headers")
    compare.add_argument("--sam3-policy", choices=("annotator-observed", "all-emitted"))
    compare.add_argument("--sam3-original-source", help="Original comparator source directory for published SAM3 cleanup")
    compare.add_argument("--sam3-cleanup-report", help="Published cleanup report (default: JSONL basename.report.json)")
    compare.add_argument("--sam3-cleanup-receipt", help="Published cleanup receipt (default: JSONL basename.receipt.json)")
    compare.add_argument("--alignment")
    compare.add_argument("--start-s", type=float)
    compare.add_argument("--end-s", type=float)
    compare.add_argument("--frame-step", type=int)
    compare.add_argument("--frame-origin", type=int)
    compare.add_argument("--source-fps", type=float)
    compare.add_argument("--roi", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    compare.add_argument("--max-distance-m", type=float)
    compare.add_argument("--similarity", choices=("linear", "gaussian"))
    compare.add_argument("--similarity-scale-m", type=float)
    compare.add_argument("--output", required=True, help="New report directory; existing paths are refused")
    compare.add_argument("--label", default="comparison")
    compare.add_argument("--note", action="append", default=[])
    inspect = sub.add_parser("inspect-deepstream", help="Read DB3 metadata without asserting GT alignment or scoring")
    inspect.add_argument("path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect-deepstream":
            from .deepstream_input import inspect_deepstream
            print(json.dumps(inspect_deepstream(args.path), indent=2, allow_nan=False))
            return 0
        requested_output = Path(args.output).expanduser().absolute()
        if requested_output.exists() or requested_output.is_symlink():
            raise ValueError(f"Output path already exists; choose a new directory: {requested_output}")
        output = requested_output.resolve()
        started, cpu_started = time.perf_counter(), time.process_time()
        entries = list(_load_manifest(args))
        config = dict(entries[0][1].get("metric_config", {}))
        for key in ("max_distance_m", "similarity", "similarity_scale_m"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        sequences = [run_sequence(entry, config) for entry, _ in entries]
        from .metrics import aggregate_results
        combined = aggregate_results([sequence["metrics"] for sequence in sequences])
        elapsed = time.perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = {"schema_version": 1, "label": args.label,
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "notes": entries[0][1].get("notes", []) + args.note,
                  "metric_config": combined["config"], "sequences": sequences, "combined": combined,
                  "runtime": {"wall_s_before_report_write": elapsed, "cpu_s": time.process_time() - cpu_started,
                              "peak_process_rss_mib": rss / (1048576 if platform.system() == "Darwin" else 1024),
                              "end_to_end_frames_per_second": combined["frame_count"] / elapsed if elapsed else None,
                              "python": platform.python_version(), "platform": platform.platform(),
                              "assignment_backend": combined.get("assignment_backend")}}
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        markdown = markdown_report(report)
        output.mkdir(parents=True, exist_ok=False)
        output.chmod(0o777)
        with (output / "report.json").open("x") as stream:
            os.fchmod(stream.fileno(), 0o777)
            stream.write(serialized)
        with (output / "report.md").open("x") as stream:
            os.fchmod(stream.fileno(), 0o777)
            stream.write(markdown)
        print(json.dumps({"output": str(output), "frame_count": combined["frame_count"],
                          "runtime": report["runtime"], "metrics": {
                              **{k: combined[k] for k in ("HOTA", "IDF1", "MOTP_m", "yaw_MAE_deg", "TP", "FP", "FN")},
                              "CLEAR_MOTA": combined.get("clear", {}).get("MOTA"),
                              "CLEAR_IDSW": combined.get("clear", {}).get("IDSW"),
                              "CLEAR_Frag": combined.get("clear", {}).get("Frag")}}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
