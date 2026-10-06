import json
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

VIDEOS = {"camera": "camera_detections.mp4", "bev": "bev.mp4", "combined": "combined.mp4"}


def jsonable(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "iub" or (value.dtype.kind == "f" and np.isfinite(value).all()):
            return value.tolist()  # no NaN or inf to replace
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def apply_promotions(buffer, promotions):
    """Give a newly promoted public ID to the buffered observations that formed its private birth streak."""
    for promotion in promotions:
        for frame in buffer:
            if not promotion.consecutive_started_timestamp_ns <= frame["timestamp_ns"] <= promotion.promotion_timestamp_ns:
                continue
            pending = [p for p in frame["pending"] if p.global_birth_private_track_id == promotion.private_track_id]
            projected = [d for d in frame["projected"] if d.global_birth_private_track_id == promotion.private_track_id]
            if not pending and not projected:
                continue
            for detection in projected:
                detection.track_id = int(promotion.public_track_id)
                detection.global_birth_pending = False
            if not any(p.track_id == promotion.public_track_id and p.global_birth_private_track_id == promotion.private_track_id for p in frame["fused"]):
                for person in pending:
                    person.global_birth_retroactively_labeled = True
                    frame["fused"].append(replace(person, members=list(person.members), track_id=int(promotion.public_track_id),
                                                  global_birth_pending=False, global_birth_confirmed=True, global_birth_retroactively_labeled=True))


def load_scene(cache_dir):
    """Scene prompt results from cache_dir/scene/camN/PROMPT.json, without masks: {camera: [(stamp, {prompt: {boxes_xyxy, scores}})]}
    sorted by stamp."""
    scene = {}
    for camera_dir in sorted((Path(cache_dir) / "scene").glob("cam*")):
        by_stamp = {}
        for path in sorted(camera_dir.glob("*.json")):
            for stamp, entry in json.loads(path.read_text()).items():
                by_stamp.setdefault(int(stamp), {})[path.stem] = {"boxes_xyxy": entry["boxes_xyxy"], "scores": entry["scores"]}
        scene[int(camera_dir.name[3:])] = sorted(by_stamp.items())
    return scene


def model_detection(detection_id, raw):
    """One SAM detection as written to frames.jsonl (2D keypoints in pixels, 3D keypoints in metres, camera frame)."""
    rounded = lambda value, digits: None if value is None else np.round(np.asarray(value, dtype=np.float64), digits)
    return {"detection_id": detection_id, "confidence": raw.confidence, "bbox_xyxy": rounded(raw.bbox_xyxy, 2),
            "detector_bbox_xyxy": rounded(raw.model_fields["detector_metadata"]["raw_bbox"], 2),
            "keypoints_2d": rounded(raw.keypoints_2d, 2), "keypoints_3d": rounded(raw.keypoints_3d, 4),
            "global_rot_zyx": rounded(raw.global_rot_zyx, 4)}


def frame_record(frame):
    stamps = [stamp for stamp, _ in frame["images"].values()]
    return jsonable({
        "frame_index": frame["frame_index"],
        "timestamp_ns": frame["timestamp_ns"],
        "present_cameras": sorted(frame["images"]),
        "source_timestamp_ns_by_camera": {str(c): stamp for c, (stamp, _) in sorted(frame["images"].items())},
        "sync_skew_ns": max(stamps) - min(stamps),
        "model_detections_by_camera": {str(c): [model_detection(f"{frame['images'][c][0]}:cam{c}:det{index}", raw) for index, raw in enumerate(raws)]
                                       for c, raws in sorted(frame["raw"].items())},
        "scene_detections_by_camera": frame["scene"],
        "projected_detections": [{
            "detection_id": d.detection_id, "camera_index": d.camera_index, "foot_pixel_uv": d.foot_pixel_uv,
            "position_ground_xy": d.ground_xy, "yaw_ground_rad": d.ground_yaw, "suppressed_duplicate": d.suppressed_duplicate,
            "camera_track_id": d.camera_track_id, "camera_track_confirmed": d.camera_track_confirmed, "track_id": d.track_id,
            "global_birth_pending": d.global_birth_pending,
        } for d in frame["projected"]],
        "fused_people": [{
            "track_id": p.track_id, "camera_indices": p.camera_indices, "position_ground_xy": p.ground_xy,
            "position_smoothed_xy": p.smoothed_xy, "yaw_ground_rad": p.ground_yaw, "global_birth_pending": p.global_birth_pending,
            "global_birth_retroactively_labeled": p.global_birth_retroactively_labeled,
        } for p in frame["fused"]],
        "tracked_people": [{
            "track_id": t.track_id, "position_ground_xy": t.position_ground_xy, "velocity_ground_xy_mps": t.velocity_ground_xy_mps,
            "position_covariance": t.position_covariance, "yaw_ground_rad": t.ground_yaw, "yaw_concentration": t.yaw_concentration,
            "spread_m": t.spread_m, "observed_this_frame": t.observed_this_frame, "stale": not t.observed_this_frame, "missed_steps": t.missed_steps,
            "missed_seconds": t.missed_seconds, "age_steps": t.age_steps, "hit_count": t.hit_count,
            "consecutive_hit_count": t.consecutive_hit_count, "consecutive_observed_seconds": t.consecutive_observed_seconds,
            "confirmed": t.confirmed, "last_observed_timestamp_ns": t.last_observed_timestamp_ns,
            "last_observed_position_ground_xy": t.last_observed_position_ground_xy, "last_observed_camera_indices": list(t.camera_indices),
        } for t in frame["tracked"]],
    })


def write_frame(frame, output_dir, jsonl, writers, fps, renderer=None):
    jsonl.write(json.dumps(frame_record(frame), sort_keys=True, allow_nan=False) + "\n")
    if renderer is None:
        return
    images = renderer.render(frame)
    for name, image in zip(VIDEOS, images):
        image = cv2.copyMakeBorder(image, 0, image.shape[0] % 2, 0, image.shape[1] % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if name not in writers:
            writers[name] = cv2.VideoWriter(str(Path(output_dir) / VIDEOS[name]), cv2.VideoWriter_fourcc(*"mp4v"), fps, image.shape[1::-1])
            cv2.imwrite(str(Path(output_dir) / f"{VIDEOS[name][:-4]}_preview.png"), images[list(VIDEOS).index(name)])
        writers[name].write(image)
    writers["last"] = images


def close_videos(output_dir, writers):
    """Release OpenCV mp4v writers, save last-frame PNGs, then re-encode each video to H.264 (crf 18) with ffmpeg."""
    last = writers.pop("last", None)
    for writer in writers.values():
        writer.release()
    if last is not None:
        for name, image in zip(VIDEOS, last):
            cv2.imwrite(str(Path(output_dir) / f"{VIDEOS[name][:-4]}_last.png"), image)
    for name in writers:
        path = Path(output_dir) / VIDEOS[name]
        temporary = path.with_suffix(".h264.mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(path), "-map", "0:v:0",
                        "-map_metadata", "0", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-profile:v", "high", "-tag:v", "avc1", "-movflags", "+faststart", "-vsync", "passthrough", "-f", "mp4",
                        str(temporary)], check=True)
        temporary.replace(path)
