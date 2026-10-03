"""Score a pipeline output directory against a reviewed GT export with gdc_atrium/tracking_evaluation.

The evaluator normally refuses predictions whose output-folder fingerprint differs from the one the GT was annotated on.
Re-runs (original or simple) are new folders by design, so this converts both sides to the evaluator's normalized
format with its own loaders (same annotator-observed/all-emitted policy) and keeps its exact capture-timestamp check.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

EVALUATOR_ROOT = next(root for root in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[1] / "gdc_atrium")
                      if (root / "tracking_evaluation").is_dir())
sys.path.insert(0, str(EVALUATOR_ROOT))
from tracking_evaluation.inputs import load_ground_truth, load_sam3_predictions  # noqa: E402

CONVENTIONS = {"coordinate_system": {"plane": "BEV", "units": "metres"},
               "orientation_convention": {"units": "radians", "zero": "+X", "positive": "counterclockwise toward +Y"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="reviewed ground_truth.json export")
    parser.add_argument("--pred", required=True, help="pipeline output directory containing frames.jsonl")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True, help="new report directory")
    parser.add_argument("--policy", default="annotator-observed", choices=("annotator-observed", "all-emitted"))
    args, passthrough = parser.parse_known_args()

    output = Path(args.output).resolve()
    work = output.with_name(output.name + ".inputs")
    work.mkdir(parents=True)
    gt_frames, gt_meta = load_ground_truth(args.gt)
    pred_frames, _ = load_sam3_predictions(args.pred, set(gt_frames), expected_fingerprint=None, prediction_policy=args.policy)
    for side, frames, path in (("gt", gt_frames, work / "gt.json"), ("pred", pred_frames, work / "pred.json")):
        path.write_text(json.dumps({"name": args.name, "fps": 20.0, **CONVENTIONS, "frames": [
            {key: f[key] for key in ("frame_index", "timestamp_s", "source_timestamp_ns", side) if f.get(key) is not None}
            for f in frames.values()]}))
    note = f"fingerprint check bypassed: predictions {Path(args.pred).resolve()} are a re-run; GT {args.gt} (source fingerprint {gt_meta.get('source_fingerprint')})"
    subprocess.run([sys.executable, "-B", "-m", "tracking_evaluation", "compare", "--gt", str(work / "gt.json"), "--pred", str(work / "pred.json"),
                    "--prediction-format", "normalized", "--name", args.name, "--output", str(output), "--note", note, *passthrough],
                   cwd=EVALUATOR_ROOT, check=True)


if __name__ == "__main__":
    main()
