"""Render a DeepStream observation CSV as a bag-time BEV diagnostic, not metrics.

Only rendering requires Pillow; parsing and causal assembly use the standard
library. Heading arrows require orient_valid=1 and a finite recorded yaw.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


ICRA = (-4.0, 14.0, -3.0, 15.0)


@dataclass(frozen=True, slots=True)
class Observation:
    stamp_ns: int
    track_id: int
    x: float
    y: float
    yaw: float | None
    score: float | None
    row: int


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": values[0], "median": statistics.median(values),
            "p95": values[math.ceil(0.95 * len(values)) - 1], "max": values[-1]}


def read_observations(path, start_ns, end_ns):
    """Keep last CSV row for equal timestamp/ID, reporting ambiguity explicitly."""
    chosen, payloads, stamps = {}, defaultdict(set), set()
    source_rows = window_rows = known_yaw = 0
    first = last = None
    signature_before = path.stat()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["stamp_ns", "track_id", "x", "y", "yaw_rad", "orient_valid", "score"]
        if reader.fieldnames != expected:
            raise ValueError("Unexpected CSV columns; run the repair/validation first")
        for row_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV at row {row_number}")
            stamp, tid = int(row["stamp_ns"]), int(row["track_id"])
            x, y = float(row["x"]), float(row["y"])
            if row["orient_valid"] not in ("0", "1"):
                raise ValueError(f"Invalid orientation flag at row {row_number}")
            yaw = float(row["yaw_rad"]) if row["orient_valid"] == "1" else None
            score = float(row["score"]) if row["score"] else None
            if not all(math.isfinite(v) for v in (x, y, yaw, score) if v is not None):
                raise ValueError(f"Nonfinite coordinate or metadata at row {row_number}")
            source_rows += 1
            first = stamp if first is None else min(first, stamp)
            last = stamp if last is None else max(last, stamp)
            if not start_ns <= stamp < end_ns:
                continue
            window_rows += 1
            known_yaw += yaw is not None
            key = (stamp, tid)
            stamps.add(stamp)
            payloads[key].add((x, y, yaw, score))
            chosen[key] = Observation(stamp, tid, x, y, yaw, score, row_number)
    signature_after = path.stat()
    if (signature_before.st_size, signature_before.st_mtime_ns) != (signature_after.st_size, signature_after.st_mtime_ns):
        raise ValueError("CSV changed during reading")
    observations = sorted(chosen.values(), key=lambda obs: (obs.stamp_ns, obs.row))
    ordered_stamps = sorted(stamps)
    per_track = defaultdict(list)
    for obs in observations:
        per_track[obs.track_id].append(obs)
    repeated_xy = repeated_pose = transitions = 0
    changed_xy_intervals, per_track_intervals = [], []
    for history in per_track.values():
        previous_changed_stamp = history[0].stamp_ns
        for prev, curr in zip(history, history[1:]):
            transitions += 1
            per_track_intervals.append((curr.stamp_ns - prev.stamp_ns) / 1e6)
            same_xy = (prev.x, prev.y) == (curr.x, curr.y)
            repeated_xy += same_xy
            repeated_pose += same_xy and prev.yaw == curr.yaw
            if not same_xy:
                changed_xy_intervals.append((curr.stamp_ns - previous_changed_stamp) / 1e6)
                previous_changed_stamp = curr.stamp_ns
    diagnostics = {
        "source_rows": source_rows, "source_first_stamp_ns": first, "source_last_stamp_ns": last,
        "first_record_bag_time_s": None if first is None else (first - start_ns) / 1e9,
        "window_rows_before_assembly": window_rows, "window_rows_with_known_heading": known_yaw,
        "window_unique_stamp_id": len(chosen), "window_unique_timestamps": len(stamps),
        "window_track_ids": sorted(per_track),
        "window_duplicate_stamp_id_extra_rows": window_rows - len(chosen),
        "window_conflicting_stamp_id_groups": sum(len(p) > 1 for p in payloads.values()),
        "unique_timestamp_step_ms": distribution([(b-a)/1e6 for a,b in zip(ordered_stamps, ordered_stamps[1:])]),
        "per_track_record_step_ms": distribution(per_track_intervals),
        "per_track_xy_change_step_ms": distribution(changed_xy_intervals),
        "consecutive_per_track_transitions": transitions,
        "transitions_with_identical_xy": repeated_xy,
        "transitions_with_identical_xy_and_heading": repeated_pose,
        "timing_caveat": "Record stamps and identical poses do not establish fresh detector cadence; stationary people and cached records are indistinguishable here.",
    }
    return observations, diagnostics


def assemble_frames(observations, start_ns, frame_count, fps, max_age_ns, bounds=ICRA):
    """Causal sample-and-hold; no future sample and no hold older than max age."""
    pending, active = iter(observations), {}
    following = next(pending, None)
    for index in range(frame_count):
        stamp = start_ns + round(index * 1e9 / fps)
        while following is not None and following.stamp_ns <= stamp:
            active[following.track_id] = following
            following = next(pending, None)
        stale = [tid for tid, obs in active.items() if stamp - obs.stamp_ns > max_age_ns]
        for tid in stale:
            del active[tid]
        xmin, xmax, ymin, ymax = bounds
        inside = sorted((obs for obs in active.values() if xmin <= obs.x <= xmax and ymin <= obs.y <= ymax),
                        key=lambda obs: obs.track_id)
        yield {"frame_index": index, "stamp_ns": stamp, "time_s": index / fps,
               "observations": inside, "active_count_before_crop": len(active)}


def make_renderer(first_record_s, fps, max_age_s, frame_count):
    from PIL import Image, ImageDraw, ImageFont
    width, height, left, top, size = 1440, 1080, 88, 134, 858
    bg, ink, muted, grid = "#101923", "#e8eef3", "#9dafbf", "#293845"
    fonts = {}
    for name, points in (("title", 34), ("section", 23), ("body", 19), ("small", 16), ("node", 17)):
        file = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if name in ("title", "section", "node") else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        fonts[name] = ImageFont.truetype(file, points)
    base = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(base)
    draw.text((40, 28), "DeepStream / dense atrium", fill=ink, font=fonts["title"])
    draw.text((40, 77), "First minute of bag time  |  ICRA world XY  |  Repaired CSV", fill=muted, font=fonts["body"])

    def point(x, y):
        return left + (x + 4) / 18 * size, top + (15 - y) / 18 * size

    for x in range(-4, 15):
        px, _ = point(x, 0)
        draw.line((px, top, px, top + size), fill="#466173" if x == 0 else grid, width=2 if x == 0 else 1)
        if x % 2 == 0:
            draw.text((px, top + size + 12), str(x), fill=muted, font=fonts["small"], anchor="mt")
    for y in range(-3, 16):
        _, py = point(0, y)
        draw.line((left, py, left + size, py), fill="#466173" if y == 0 else grid, width=2 if y == 0 else 1)
        if y % 2 == 1:
            draw.text((left - 15, py), str(y), fill=muted, font=fonts["small"], anchor="rm")
    draw.rectangle((left, top, left + size, top + size), outline="#607989", width=2)
    draw.text((left + size / 2, 1034), "World X (m)", fill=ink, font=fonts["body"], anchor="mt")
    draw.text((left, top - 34), "World Y (m)", fill=ink, font=fonts["body"])
    draw.rounded_rectangle((990, 134, 1400, 990), 16, fill="#192633", outline="#304352")
    draw.text((1016, 330), "ICRA range", fill=ink, font=fonts["section"])
    draw.text((1016, 370), "X: -4 to 14 m   Y: -3 to 15 m", fill=muted, font=fonts["small"])
    draw.text((1016, 430), "Legend", fill=ink, font=fonts["section"])
    draw.ellipse((1020, 483, 1038, 501), fill="#57d0cd")
    draw.line((1029, 492, 1052, 478), fill="#57d0cd", width=3)
    draw.text((1067, 480), "Recorded heading", fill=ink, font=fonts["body"])
    draw.ellipse((1020, 532, 1038, 550), outline="#57d0cd", width=3)
    draw.text((1067, 529), "Heading unavailable", fill=ink, font=fonts["body"])
    draw.text((1016, 590), "Preview assembly", fill=ink, font=fonts["section"])
    lines = [f"{fps:g} fps display; causal last record", f"per ID, age at most {max_age_s:g} seconds.",
             "Same timestamp + ID: last CSV row.", "No interpolation or inferred headings.",
             "", "Record rate is not detector freshness.", "This video is not metric alignment."]
    for offset, text_line in enumerate(lines):
        draw.text((1016, 634 + offset * 30), text_line, fill=muted, font=fonts["small"])
    draw.text((1016, 883), "Recording starts", fill=ink, font=fonts["section"])
    draw.text((1016, 925), f"Bag +{first_record_s:.3f} s", fill="#efb663", font=fonts["body"])

    def render(frame):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        draw.text((1016, 164), f"BAG  {frame['time_s']:06.2f} s", fill=ink, font=fonts["title"])
        draw.text((1016, 220), f"Display frame {frame['frame_index']:04d} / {frame_count-1}", fill=muted, font=fonts["body"])
        nodes = frame["observations"]
        draw.text((1016, 268), f"{len(nodes)} IDs visible in ICRA", fill="#57d0cd", font=fonts["section"])
        palette = ("#57d0cd", "#f3bd66", "#a998f5", "#ed8eac", "#81c985", "#80bafa")
        label_boxes = []
        for obs in nodes:
            px, py = point(obs.x, obs.y)
            color = palette[obs.track_id % len(palette)]
            radius = 6
            if obs.yaw is None:
                draw.ellipse((px-radius, py-radius, px+radius, py+radius), outline=color, width=3)
            else:
                dx, dy = 26 * math.cos(obs.yaw), -26 * math.sin(obs.yaw)
                tip = (px+dx, py+dy)
                draw.line((px, py, *tip), fill=color, width=3)
                normx, normy = dx/26, dy/26
                draw.polygon((tip, (tip[0]-8*normx+4*normy, tip[1]-8*normy-4*normx),
                                   (tip[0]-8*normx-4*normy, tip[1]-8*normy+4*normx)), fill=color)
                draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=color)
            label = f"T{obs.track_id}"
            tw = draw.textbbox((0, 0), label, font=fonts["node"])[2]
            candidates = [(px+10,py-21), (px+10,py+5), (px-tw-14,py-21), (px-tw-14,py+5),
                          (px+10,py-42), (px+10,py+26)]
            def penalty(candidate):
                x, y = candidate
                box = (x-2, y-1, x+tw+3, y+21)
                overlap = sum(max(0,min(box[2],b[2])-max(box[0],b[0])) * max(0,min(box[3],b[3])-max(box[1],b[1])) for b in label_boxes)
                outside = 10000 if box[0]<left or box[2]>left+size or box[1]<top or box[3]>top+size else 0
                return overlap + outside
            lx, ly = min(candidates, key=penalty)
            box = (lx-2, ly-1, lx+tw+3, ly+21)
            label_boxes.append(box)
            draw.rounded_rectangle(box, 3, fill=bg)
            draw.text((lx, ly), label, fill=color, font=fonts["node"])
        if not nodes:
            before_recording = frame["time_s"] < first_record_s
            if before_recording:
                heading, detail = "NO OBSERVATIONS RECORDED", f"First CSV observation at bag +{first_record_s:.3f} s"
            elif not frame["active_count_before_crop"]:
                heading, detail = "NO RECENT RECORDS", f"No records in the preceding {max_age_s:g} seconds"
            else:
                heading, detail = "NO RECENT RECORDS INSIDE ICRA", "This is recorded-data coverage, not an empty-scene claim"
            draw.rounded_rectangle((170, 479, 864, 621), 12, fill="#192633", outline="#607989")
            draw.text((517, 510), heading, fill="#efb663", font=fonts["section"], anchor="mt")
            draw.text((517, 559), detail, fill=muted, font=fonts["small"], anchor="mt")
        return image
    return render


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--bag-start-ns", required=True, type=int)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-age-s", type=float, default=0.15)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.fps <= 0 or args.max_age_s < 0 or not all(math.isfinite(v) for v in (args.seconds,args.fps,args.max_age_s)):
        parser.error("seconds/fps must be positive and max-age-s nonnegative, all finite")
    frame_count = round(args.seconds * args.fps)
    if not math.isclose(frame_count / args.fps, args.seconds):
        parser.error("seconds times fps must be an integer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.chmod(0o777)
    video = args.output_dir / "dense_deepstream_first_60s.mp4"
    report_path = args.output_dir / "preview_report.json"
    if any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty to preserve prior artifacts")
    source_hash = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    observations, report = read_observations(args.csv, args.bag_start_ns, args.bag_start_ns + round(args.seconds*1e9))
    if not observations:
        raise ValueError("No CSV records in the requested bag-time interval")
    report.update({"source_csv": str(args.csv.absolute()), "source_sha256": source_hash,
                   "bag_start_ns": args.bag_start_ns, "duration_s": args.seconds,
                   "render_fps": args.fps, "frame_count": frame_count,
                   "sample_times_s": {"first": 0, "last": (frame_count-1)/args.fps},
                   "bounds_xy_inclusive": list(ICRA), "max_record_age_s": args.max_age_s,
                   "same_stamp_id_policy": "Last row in repaired CSV order; visualization only, not endorsed for metrics",
                   "sampling": "Causal latest record per ID within maximum age; preserve startup gap; no interpolation",
                   "heading_policy": "Draw arrows only for orient_valid=1 with finite recorded yaw; no inference"})
    renderer = make_renderer(report["first_record_bag_time_s"], args.fps, args.max_age_s, frame_count)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "1440x1080", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
               "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    report["ffmpeg_command"] = command
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    contact = []
    target_seconds = (0, 21.6, 25, 30, 40, 50)
    contact_indices = {round(t * args.fps) for t in target_seconds if t < args.seconds}
    visible_counts, empty_frames = [], []
    samples = []
    try:
        for frame in assemble_frames(observations, args.bag_start_ns, frame_count, args.fps, round(args.max_age_s*1e9)):
            image = renderer(frame)
            process.stdin.write(image.tobytes())
            visible_counts.append(len(frame["observations"]))
            if not frame["observations"]:
                empty_frames.append(frame["frame_index"])
            if frame["frame_index"] in contact_indices:
                image.save(args.output_dir / f"sample_{frame['time_s']:05.2f}s.png")
                contact.append(image.resize((720, 540)))
                samples.append({"frame_index": frame["frame_index"], "time_s": frame["time_s"],
                                "ids": [obs.track_id for obs in frame["observations"]],
                                "known_heading_count": sum(obs.yaw is not None for obs in frame["observations"])})
            if frame["frame_index"] % 200 == 0:
                print(f"Rendered {frame['frame_index']}/{frame_count}", flush=True)
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    from PIL import Image
    contact_sheet = Image.new("RGB", (1440, 540 * math.ceil(len(contact)/2)), "#101923")
    for index, im in enumerate(contact):
        contact_sheet.paste(im, ((index % 2)*720, (index // 2)*540))
    contact_sheet.save(args.output_dir / "contact_sheet.jpg", quality=93)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
                        "-of", "json", str(video)], text=True))
    stream = probe["streams"][0]
    if int(stream["nb_read_frames"]) != frame_count or not math.isclose(float(stream["duration"]), args.seconds, abs_tol=1e-5):
        raise RuntimeError(f"Unexpected encoded frame count or duration: {probe}")
    report.update({"ffprobe": probe, "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                   "visible_id_counts": distribution(visible_counts), "frames_without_visible_records": empty_frames,
                   "sample_frames": samples})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for artifact in args.output_dir.iterdir():
        artifact.chmod(0o777)
    print(json.dumps({"video": str(video), "report": str(report_path), "ffprobe": probe}, indent=2))


if __name__ == "__main__":
    main()
