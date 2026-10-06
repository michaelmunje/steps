"""Read the output of extract_videos.py: same interface as bag.read_bag_index / bag.iter_group_images, but from per-camera
rectified videos (images come out already rectified). play() replays them as a live frame source for realtime.py."""
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def read_index(frames_dir, max_groups=None):
    index = json.loads((Path(frames_dir) / "frames.json").read_text())
    camera_infos = {int(c): {k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v) for k, v in info.items()}
                    for c, info in index["camera_infos"].items()}
    groups = [(timestamp, {int(c): stamp for c, stamp in frames.items()}) for timestamp, frames in index["groups"]]
    return camera_infos, groups[:max_groups]


def iter_group_images(frames_dir, groups):
    """Yield (timestamp, {camera: (stamp, rectified bgr)}) per group; each camera video holds its images in timeline order."""
    videos = {}
    for timestamp, frames in groups:
        images = {}
        for camera, stamp in sorted(frames.items()):
            if camera not in videos:
                videos[camera] = cv2.VideoCapture(str(Path(frames_dir) / f"cam{camera}.mp4"))
            ok, image = videos[camera].read()
            if not ok:
                raise RuntimeError(f"cam{camera}.mp4 ended before stamp {stamp}")
            images[camera] = (stamp, image)
        yield timestamp, images


def play(frames_dir, groups, buffers, on_frame, paced):
    """Replay the videos as live cameras: on_frame(frame_index, timestamp_ns, {camera: stamp}, pixels) per group, at its recorded time
    (paced) or as soon as on_frame returns. pixels is the next of `buffers` ((cameras, H, W, 3) uint8 arrays in pinned memory, reused
    round-robin), which each camera video decodes straight into; decoding runs ahead by up to len(buffers) - 4 groups."""
    videos = {c: cv2.VideoCapture(str(Path(frames_dir) / f"cam{c}.mp4")) for c in sorted({c for _, group in groups for c in group})}
    pool = ThreadPoolExecutor(len(videos))
    decoded, done = queue.Queue(maxsize=len(buffers) - 4), object()

    def read(camera, pixels):
        if not videos[camera].read(pixels[camera])[0]:
            raise RuntimeError(f"cam{camera}.mp4 ended early")

    def decode():
        for frame_index, (timestamp_ns, group) in enumerate(groups):
            pixels = buffers[frame_index % len(buffers)]
            list(pool.map(read, sorted(group), [pixels] * len(group)))
            decoded.put((frame_index, timestamp_ns, group, pixels))
        decoded.put(done)

    threading.Thread(target=decode, daemon=True).start()
    while paced and not decoded.full():
        time.sleep(0.01)
    start_s, first_ns = time.perf_counter(), groups[0][0]
    while (item := decoded.get()) is not done:
        if paced:
            time.sleep(max(0.0, start_s + (item[1] - first_ns) / 1e9 - time.perf_counter()))
        on_frame(*item)
