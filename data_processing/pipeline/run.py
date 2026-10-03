"""Pass 2 (CPU): videos + SAM cache (infer.py) -> floor projection -> multi-camera tracking -> frames.jsonl + videos."""
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

import frames
import ground
import output
import sam
import track


def run(frames_dir, groups, cameras, cache_dir, output_dir, tracking, render, fixed_lag_s):
    """tracking: the config's `tracking` section; render: also write the videos; fixed_lag_s: output delay that lets confirmed births
    relabel their earlier frames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = track.MultiCameraTracker(**tracking)

    deltas = np.diff(np.asarray([timestamp for timestamp, _ in groups], dtype=np.int64))
    fps = float(np.clip(1e9 / np.median(deltas[deltas > 0]) if (deltas > 0).any() else 10.0, 0.1, 120.0))
    renderer = None
    if render:
        import render as render_module
        renderer = render_module.Renderer(cameras)

    scene, scene_next, latest_scene = output.load_scene(cache_dir), defaultdict(int), {}
    frame_of_stamp = {(c, stamp): index for index, (_, group) in enumerate(groups) for c, stamp in group.items()}
    buffer, writers = deque(), {}
    seconds, clock = defaultdict(float), time.perf_counter()

    def lap(stage):
        nonlocal clock
        now = time.perf_counter()
        seconds[stage] += now - clock
        clock = now

    with open(output_dir / "frames.jsonl", "w") as jsonl:
        for frame_index, (timestamp_ns, images) in enumerate(frames.iter_group_images(frames_dir, groups)):
            lap("video_decode")
            raw_by_camera, projected = {}, []
            for camera_index, (stamp, image) in sorted(images.items()):
                results = scene.get(camera_index, [])
                while scene_next[camera_index] < len(results) and results[scene_next[camera_index]][0] <= stamp:  # newest scene result so far
                    scene_stamp, prompts = results[scene_next[camera_index]]
                    latest_scene[camera_index] = {"source_timestamp_ns": scene_stamp,
                                                  "source_frame_index": frame_of_stamp.get((camera_index, scene_stamp)), "prompts": prompts}
                    scene_next[camera_index] += 1
                raw_by_camera[camera_index] = sam.read_cache(sam.cache_path(cache_dir, camera_index, stamp), sam.fingerprint(image, cameras[camera_index]["K"]))
                lap("sam_cache_read")
                projected += ground.project_detections(raw_by_camera[camera_index], cameras[camera_index], stamp)
                lap("quality_projection")

            fused = tracker.update(projected, timestamp_ns)
            lap("tracking")

            output.apply_promotions(buffer, tracker.last_global_birth_promotions)
            buffer.append({"frame_index": frame_index, "timestamp_ns": timestamp_ns, "images": images, "raw": raw_by_camera,
                           "scene": {str(c): latest_scene[c] for c in sorted(images) if c in latest_scene},
                           "projected": projected, "fused": fused, "pending": list(tracker.last_pending_birth_observations),
                           "tracked": tracker.estimates()})
            while buffer and buffer[0]["timestamp_ns"] <= timestamp_ns - int(round(fixed_lag_s * 1e9)):
                output.write_frame(buffer.popleft(), output_dir, jsonl, writers, fps, renderer)
            lap("output_render_json")
            if frame_index % 10 == 0:
                print(f"[{frame_index + 1:4d}/{len(groups)}] projected={len(projected):2d} fused={len(fused):2d}", flush=True)
        while buffer:
            output.write_frame(buffer.popleft(), output_dir, jsonl, writers, fps, renderer)
    lap("output_render_json")
    output.close_videos(output_dir, writers)
    lap("video_encode")
    per_frame = {stage: total / len(groups) for stage, total in seconds.items()}
    (output_dir / "timing.json").write_text(json.dumps({"frames": len(groups), "cache_dir": str(Path(cache_dir).resolve()), "seconds_per_frame": per_frame, "total_seconds": dict(seconds)}, indent=1))
    print("seconds per frame: " + ", ".join(f"{stage} {value:.3f}" for stage, value in per_frame.items()), flush=True)

