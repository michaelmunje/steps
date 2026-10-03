"""Rosbag -> one losslessly encoded, rectified video per camera + frames.json (recorded CameraInfo, sync timeline, per-camera stamps)."""
import json
import subprocess
import time
from pathlib import Path

import bag
import rectify


def run(bag_path, cameras_dir, extrinsics_file, output_dir, video_fps, sync_tolerance_ms, max_groups=None):
    """Parameters after output_dir: the config's `extract` section."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    camera_infos, groups = bag.read_bag_index(bag_path, int(round(sync_tolerance_ms * 1e6)), max_groups=max_groups)
    cameras = rectify.load_cameras(cameras_dir, camera_infos, extrinsics_file)
    encoders, stamps = {}, {camera: [] for camera in cameras}
    for _, images in bag.iter_group_images(bag_path, groups):
        for camera_index, (stamp, image) in sorted(images.items()):
            image = rectify.rectify(image, cameras[camera_index])
            if camera_index not in encoders:
                encoders[camera_index] = subprocess.Popen(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                     "-s", f"{image.shape[1]}x{image.shape[0]}", "-r", str(video_fps), "-i", "-",
                     "-c:v", "libx264rgb", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "bgr24", str(output_dir / f"cam{camera_index}.mp4")],
                    stdin=subprocess.PIPE)
            encoders[camera_index].stdin.write(image.tobytes())
            stamps[camera_index].append(stamp)
    for encoder in encoders.values():
        encoder.stdin.close()
        encoder.wait()
    info = {str(c): {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in camera_infos[c].items()} for c in camera_infos}
    (output_dir / "frames.json").write_text(json.dumps({"fps": video_fps, "camera_infos": info,
                                                        "groups": [[t, {str(c): s for c, s in f.items()}] for t, f in groups],
                                                        "stamps": {str(k): v for k, v in stamps.items()}}))
    print(f"extracted {sum(map(len, stamps.values()))} images in {time.perf_counter() - started:.1f} s", flush=True)

