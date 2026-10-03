"""GDC atrium pipeline: (rosbag ->) one video per camera -> pass 1 (GPU: SAM3 people + scene prompts, SAM 3D Body -> SAM cache)
-> pass 2 (CPU: floor projection, multi-camera tracking -> frames.jsonl + videos).

  python pipeline/pipeline.py --config config/pipeline.yaml --mp4-dir data/videos/lossless --output-dir outputs/myrun
  python pipeline/pipeline.py --config config/pipeline.yaml --rosbag-dir path/to/bag --output-dir outputs/myrun [--extract-only]

--mp4-dir: cam0..5.mp4 + frames.json (cameras, frame timeline, per-camera stamps), as written from a rosbag.
--rosbag-dir: a ROS 2 bag directory (or one .mcap file); its videos are written to OUTPUT/videos first.
The SAM cache is OUTPUT/cache; a rerun reuses it and skips pass 1 when it is complete. All settings: the config file."""
import argparse
import os
import sys
from pathlib import Path

import yaml


def load_config(path):
    """The YAML config, with its paths resolved relative to the config file."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    config["paths"] = {name: (path.parent / value).resolve() for name, value in config["paths"].items()}
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mp4-dir", help="cam0..5.mp4 + frames.json")
    source.add_argument("--rosbag-dir", help="ROS 2 bag directory or .mcap file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extract-only", action="store_true", help="with --rosbag-dir: only write OUTPUT/videos (e.g. before a transfer)")
    parser.add_argument("--max-frames", type=int, help="only the first N frames (quick tests)")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    os.environ["HF_HUB_OFFLINE"] = "1"  # all weights are local files
    sys.path.insert(0, str(paths["sam3d_body_code"]))
    import extract_videos, frames, infer, rectify, run  # noqa: E401 (needs sam3d_body_code on the path)

    output_dir = Path(args.output_dir)
    mp4_dir = args.mp4_dir
    if args.rosbag_dir:
        mp4_dir = output_dir / "videos"
        extract_videos.run(args.rosbag_dir, paths["cameras"], paths["extrinsics"], mp4_dir, **config["extract"], max_groups=args.max_frames)
        if args.extract_only:
            return

    camera_infos, groups = frames.read_index(mp4_dir, max_groups=args.max_frames)
    cameras = rectify.load_cameras(paths["cameras"], camera_infos, paths["extrinsics"])
    cache_dir = output_dir / "cache"
    if not infer.cache_complete(groups, cameras, cache_dir, config["detection"]["scene_prompts"]):
        infer.run(mp4_dir, groups, cameras, cache_dir, paths["sam3d_body_checkpoint"], paths["mhr_model"], paths["sam3_checkpoint"],
                  paths["dinov3_code"], **config["detection"])
    run.run(mp4_dir, groups, cameras, cache_dir, output_dir, config["tracking"], **config["output"])


if __name__ == "__main__":
    main()
