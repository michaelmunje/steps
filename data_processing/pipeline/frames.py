"""Read the output of extract_videos.py: same interface as bag.read_bag_index / bag.iter_group_images, but from per-camera
rectified videos (images come out already rectified)."""
import json
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
