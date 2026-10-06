"""Real-time pipeline: six cameras -> YOLO person boxes -> floor positions -> multi-camera tracking -> frames.jsonl, one frame at a time.

  python pipeline/realtime.py --config config/pipeline.yaml --mp4-dir data/videos/lossless --output-dir outputs/fast [--no-pace]

The frame source replays extracted videos at their recorded rate (--no-pace: as fast as the pipeline runs). It calls
on_frame(frame_index, timestamp_ns, {camera: stamp}, pixels) per synchronized frame, from its own thread; a live ROS source would do
the same with rectified images. Only the newest frame waits to be processed: a frame not picked up before the next one arrives is
dropped, and its output holds the tracks' predicted positions. The GPU detects people in all cameras of a frame at once while the
previous frame is projected, tracked and published. Settings: the config's `realtime` section and `tracking` (with its overrides)."""
import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

import frames
import ground
import output
import rectify
import track
import yolo
from pipeline import load_config


class LatestFrame:
    """The newest frame from the source, with the frames it replaced (dropped). lossless: the source waits instead of replacing."""

    def __init__(self, lossless):
        self.lossless, self.frame, self.dropped, self.closed = lossless, None, [], False
        self.changed = threading.Condition()

    def put(self, frame_index, timestamp_ns, stamps, pixels):
        with self.changed:
            while self.lossless and self.frame is not None:
                self.changed.wait()
            if self.frame is not None:
                self.dropped.append(self.frame)
            self.frame = (frame_index, timestamp_ns, stamps, pixels, time.perf_counter())
            self.changed.notify_all()

    def close(self):
        with self.changed:
            while self.frame is not None:
                self.changed.wait()
            self.closed = True
            self.changed.notify_all()

    def take(self):
        """(frame, frames dropped before it), or (None, []) once the source is closed."""
        with self.changed:
            while self.frame is None and not self.closed:
                self.changed.wait()
            frame, dropped, self.frame, self.dropped = self.frame, self.dropped, None, []
            self.changed.notify_all()
            return frame, dropped


class JsonlPublisher:
    """Writes each published frame as one frames.jsonl line, on its own thread so file I/O never holds up tracking."""

    def __init__(self, path):
        self.lines = queue.Queue()
        self.thread = threading.Thread(target=self.write, args=(path,))
        self.thread.start()

    def write(self, path):
        with open(path, "w") as jsonl:
            while (line := self.lines.get()) is not None:
                jsonl.write(line)

    def publish(self, frame):
        self.lines.put(json.dumps(output.frame_record(frame), sort_keys=True, allow_nan=False) + "\n")

    def close(self):
        self.lines.put(None)
        self.thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mp4-dir", required=True, help="cam0..5.mp4 + frames.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-pace", dest="pace", action="store_false", help="replay as fast as the pipeline runs (throughput test)")
    parser.add_argument("--max-frames", type=int, help="only the first N frames (quick tests)")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["realtime"]
    camera_infos, groups = frames.read_index(args.mp4_dir, max_groups=args.max_frames)
    cameras = rectify.load_cameras(config["paths"]["cameras"], camera_infos, config["paths"]["extrinsics"])
    tracking = {**config["tracking"], **settings["tracking"]}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = yolo.load_detector(config["paths"]["yolo_detector"])
    shape = (max(cameras) + 1, cameras[0]["height"], cameras[0]["width"], 3)
    buffers = [torch.zeros(shape, dtype=torch.uint8).pin_memory().numpy() for _ in range(16)]
    timestamp_ns, images = next(frames.iter_group_images(args.mp4_dir, groups[:1]))  # warm-up: first calls are slow
    for c, (_, image) in images.items():
        buffers[0][c] = image
    for _ in range(3):
        people = yolo.find_people(detector, buffers[0], settings["image_size"], settings["min_confidence"])
    projected = [p for c, (stamp, _) in images.items()
                 for p in ground.project_boxes(*people[c], cameras[c], stamp, settings["foot_raise"])[1]]
    track.MultiCameraTracker(**tracking).update(projected, timestamp_ns)

    latest = LatestFrame(lossless=not args.pace)
    detected, latencies_s = queue.Queue(maxsize=1), []
    tracker = track.MultiCameraTracker(**tracking)
    publisher = JsonlPublisher(output_dir / "frames.jsonl")

    def replay():
        frames.play(args.mp4_dir, groups, buffers, latest.put, args.pace)
        latest.close()

    def track_and_publish():
        last_update_ns = groups[0][0]
        while (item := detected.get()) is not None:
            (frame_index, timestamp_ns, stamps, _, arrival_s), people = item
            images = {c: (stamp, None) for c, stamp in stamps.items()}
            if people is None:  # dropped frame: the tracks moved on by their velocity
                estimates, dt_s = tracker.estimates(), (timestamp_ns - last_update_ns) / 1e9
                for estimate in estimates:
                    estimate.position_ground_xy = estimate.position_ground_xy + dt_s * estimate.velocity_ground_xy_mps
                    estimate.observed_this_frame = False
                publisher.publish({"frame_index": frame_index, "timestamp_ns": timestamp_ns, "images": images, "raw": {}, "scene": {},
                                   "projected": [], "fused": [], "tracked": estimates})
                continue
            raw_by_camera, projected = {}, []
            for c, stamp in stamps.items():
                raw_by_camera[c], camera_projected = ground.project_boxes(*people[c], cameras[c], stamp, settings["foot_raise"])
                projected += camera_projected
            fused = tracker.update(projected, timestamp_ns)
            last_update_ns = timestamp_ns
            publisher.publish({"frame_index": frame_index, "timestamp_ns": timestamp_ns, "images": images, "raw": raw_by_camera,
                               "scene": {}, "projected": projected, "fused": fused, "tracked": tracker.estimates()})
            latencies_s.append(time.perf_counter() - arrival_s)

    threads = [threading.Thread(target=replay, daemon=True), threading.Thread(target=track_and_publish)]
    for thread in threads:
        thread.start()
    start_s, processed, dropped = time.perf_counter(), 0, 0
    while True:
        frame, dropped_frames = latest.take()
        if frame is None:
            break
        for dropped_frame in dropped_frames:
            detected.put((dropped_frame, None))
        dropped += len(dropped_frames)
        detected.put((frame, yolo.find_people(detector, frame[3], settings["image_size"], settings["min_confidence"])))
        processed += 1
    detected.put(None)
    threads[1].join()
    publisher.close()

    seconds = time.perf_counter() - start_s
    latency_ms = 1000 * np.asarray(latencies_s)
    timing = {"frames": len(groups), "processed": processed, "dropped": dropped, "paced": args.pace,
              "frames_per_second": processed / seconds,
              "latency_ms": {"median": float(np.median(latency_ms)), "p95": float(np.percentile(latency_ms, 95)),
                             "p99": float(np.percentile(latency_ms, 99)), "max": float(latency_ms.max())}}
    (output_dir / "timing.json").write_text(json.dumps(timing, indent=1))
    print(f"{processed / seconds:.1f} frames/s, {dropped} dropped, latency median {timing['latency_ms']['median']:.0f} ms "
          f"(p99 {timing['latency_ms']['p99']:.0f} ms)", flush=True)


if __name__ == "__main__":
    main()
