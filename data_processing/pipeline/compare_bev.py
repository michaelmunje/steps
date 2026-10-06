"""Side-by-side comparison video: the six cameras on top, the confirmed tracks of two runs in bird's-eye view below.
  python pipeline/compare_bev.py --mp4-dir data/videos/lossless --left outputs/A "Full Pipeline" --right outputs/B "Fast Pipeline" \
      --left-fps 0.23 --right-fps 30 --output outputs/compare [--left-note TEXT] [--right-note TEXT]
Writes OUTPUT/compare.mp4 (H.264, 20 fps) and OUTPUT/compare_<start>-<end>s.gif, one 720-pixel 10 fps clip per 10 seconds."""
import argparse
import colorsys
import json
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import yaml

EXTENT = (-4.0, 14.0, -3.0, 15.0)  # x_min, x_max, y_min, y_max (m): the evaluated region
TRAIL_S = 4.0
FONT = cv2.FONT_HERSHEY_SIMPLEX


def track_color(track_id):
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.85)
    return int(b * 255), int(g * 255), int(r * 255)


class Bev:
    def __init__(self, title, note, camera_xy, fps, size=720):
        self.size, self.title, self.note, self.fps = size, title, note, fps
        self.ppm = (size - 40) / max(EXTENT[1] - EXTENT[0], EXTENT[3] - EXTENT[2])
        self.base = np.full((size, size, 3), 246, np.uint8)
        for x in range(int(EXTENT[0]), int(EXTENT[1]) + 1):
            cv2.line(self.base, self.px((x, EXTENT[2])), self.px((x, EXTENT[3])), (222, 222, 222) if x % 5 else (200, 200, 200), 1)
        for y in range(int(EXTENT[2]), int(EXTENT[3]) + 1):
            cv2.line(self.base, self.px((EXTENT[0], y)), self.px((EXTENT[1], y)), (222, 222, 222) if y % 5 else (200, 200, 200), 1)
        for index, position in camera_xy.items():
            point = self.px(position)
            cv2.drawMarker(self.base, point, (60, 60, 60), cv2.MARKER_TRIANGLE_UP, 14, 2)
            cv2.putText(self.base, f"cam{index}", (point[0] + 8, point[1] + 4), FONT, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        self.history = defaultdict(deque)  # track id -> (timestamp, pixel)

    def px(self, xy):
        return (int(round(20 + (xy[0] - EXTENT[0]) * self.ppm)), int(round(20 + (EXTENT[3] - xy[1]) * self.ppm)))

    def draw(self, frame):
        now = frame["timestamp_ns"]
        current = {t["track_id"]: t["position_ground_xy"] for t in frame["tracked_people"] if t["confirmed"]}
        for track_id, xy in current.items():
            self.history[track_id].append((now, self.px(xy)))
        panel = self.base.copy()
        for track_id, points in list(self.history.items()):
            while points and points[0][0] < now - TRAIL_S * 1e9:
                points.popleft()
            if not points:
                del self.history[track_id]
                continue
            color = track_color(track_id)
            for (t0, p0), (t1, p1) in zip(points, list(points)[1:]):
                fade = 1.0 - (now - t1) / (TRAIL_S * 1e9)
                cv2.line(panel, p0, p1, tuple(int(246 + (c - 246) * fade) for c in color), 2, cv2.LINE_AA)
            if track_id in current:
                cv2.circle(panel, points[-1][1], 5, color, -1, cv2.LINE_AA)
        cv2.rectangle(panel, (0, 0), (self.size, 30), (40, 40, 40), -1)
        cv2.putText(panel, f"{self.title}   {len(current)} people", (10, 21), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        label = f"{self.fps:g} FPS"
        width = cv2.getTextSize(label, FONT, 0.6, 2)[0][0]
        cv2.putText(panel, label, (self.size - width - 10, 21), FONT, 0.6, (120, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, self.note, (10, self.size - 10), FONT, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--left", nargs=2, required=True, metavar=("RUN_DIR", "TITLE"))
    parser.add_argument("--right", nargs=2, required=True, metavar=("RUN_DIR", "TITLE"))
    parser.add_argument("--left-note", default="")
    parser.add_argument("--right-note", default="")
    parser.add_argument("--left-fps", type=float, required=True, help="end-to-end frames per second, shown in the title bar")
    parser.add_argument("--right-fps", type=float, required=True)
    parser.add_argument("--extrinsics", default="config/extrinsics_8_14.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    extrinsics = yaml.safe_load(Path(args.extrinsics).read_text())["cameras"]
    cameras = {int(name[3:]): np.asarray(value["matrix_row_major"], dtype=np.float64).reshape(3, 4)[:2, 3]
               for name, value in extrinsics.items()}
    runs = [[json.loads(line) for line in open(Path(run) / "frames.jsonl")] for run, _ in (args.left, args.right)]
    bevs = [Bev(args.left[1], args.left_note, cameras, args.left_fps), Bev(args.right[1], args.right_note, cameras, args.right_fps)]
    index = json.loads((Path(args.mp4_dir) / "frames.json").read_text())
    videos = {int(c): cv2.VideoCapture(str(Path(args.mp4_dir) / f"cam{c}.mp4")) for c in index["camera_infos"]}
    tiles = {c: np.zeros((360, 480, 3), np.uint8) for c in videos}

    temporary = output / "compare.mp4v.mp4"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (1440, 1440))
    count = min(len(runs[0]), len(runs[1]), len(index["groups"]))
    for frame_number in range(count):
        _, group = index["groups"][frame_number]
        for c in sorted(int(k) for k in group):
            ok, image = videos[c].read()
            if ok:
                tiles[c] = cv2.resize(image, (480, 360), interpolation=cv2.INTER_AREA)
                cv2.putText(tiles[c], f"cam{c}", (8, 22), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        mosaic = np.vstack([np.hstack([tiles[c] for c in (0, 1, 2)]), np.hstack([tiles[c] for c in (3, 4, 5)])])
        panels = [bev.draw(run[frame_number]) for bev, run in zip(bevs, runs)]
        writer.write(np.vstack([mosaic, np.hstack(panels)]))
    writer.release()
    ffmpeg = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    subprocess.run(ffmpeg + ["-i", str(temporary), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                             str(output / "compare.mp4")], check=True)
    temporary.unlink()
    duration_s = count / 20.0
    for start_s in range(0, int(np.ceil(duration_s)), 10):
        clip = output / f"compare_{start_s}-{min(start_s + 10, int(round(duration_s)))}s.gif"
        subprocess.run(ffmpeg + ["-ss", str(start_s), "-t", "10", "-i", str(output / "compare.mp4"), "-loop", "0", "-vf",
                                 "fps=10,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
                                 "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(clip)], check=True)
    print(f"wrote {output / 'compare.mp4'} ({count} frames) and its 10 s gif clips", flush=True)


if __name__ == "__main__":
    main()
