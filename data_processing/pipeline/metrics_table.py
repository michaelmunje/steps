"""Print scores and seconds per frame for evaluator reports: python3 metrics_table.py outputs/eval/NAME_POLICY ...
Detection: precision, recall, F1. Tracking: IDF1, HOTA, CLEAR MOTA, ID switches, fragmentations. Accuracy of matched people:
position error (cm) and yaw error (deg). Time: SAM = pass 1 (cache timing.json, without model loading), post-SAM = pass 2
(the run's timing.json), total = both."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def times(name):
    """(SAM, post-SAM, total) seconds per frame for outputs/NAME, or None where a timing.json is missing."""
    run_timing = ROOT / "outputs" / name / "timing.json"
    if not run_timing.is_file():
        return None, None, None
    run = json.load(open(run_timing))
    if "latency_ms" in run:  # realtime.py: one end-to-end number
        return None, None, 1.0 / run["frames_per_second"]
    post = sum(run["seconds_per_frame"].values())
    cache_timing = Path(run["cache_dir"]) / "timing.json"
    if not cache_timing.is_file():
        return None, post, None
    sam = sum(v for k, v in json.load(open(cache_timing))["seconds_per_frame"].items() if not k.startswith("load_"))
    return sam, post, sam + post


header = ("run", "Precision", "Recall", "F1", "IDF1", "HOTA", "MOTA", "ID sw", "Frag", "pos cm", "yaw deg", "SAM s", "postSAM s", "total s")
rows = []
for directory in sys.argv[1:]:
    c = json.load(open(Path(directory) / "report.json"))["combined"]
    f1 = 2 * c["precision"] * c["recall"] / (c["precision"] + c["recall"])
    cells = [f"{100 * v:.2f}%" for v in (c["precision"], c["recall"], f1, c["IDF1"], c["HOTA"], c["clear"]["MOTA"])]
    cells += [str(c["clear"]["IDSW"]), str(c["clear"]["Frag"]), f"{100 * c['MOTP_m']:.2f}", f"{c['yaw_MAE_deg']:.2f}" if c['yaw_MAE_deg'] is not None else "-"]
    cells += [f"{v:.2f}" if v is not None else "-" for v in times(Path(directory).name.rsplit("_", 1)[0])]
    rows.append([Path(directory).name] + cells)
width = max(len(r[0]) for r in rows + [list(header)])
print(f"{header[0]:<{width}} " + " ".join(f"{h:>9}" for h in header[1:]))
for row in rows:
    print(f"{row[0]:<{width}} " + " ".join(f"{cell:>9}" for cell in row[1:]))
