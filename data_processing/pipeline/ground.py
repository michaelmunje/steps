from dataclasses import dataclass

import numpy as np

from rectify import pixel_to_ground, sam_yaw_to_ground


@dataclass
class ProjectedDetection:
    camera_index: int
    timestamp_ns: int
    raw: object
    foot_pixel_uv: np.ndarray
    ground_xy: np.ndarray
    ground_yaw: float = None
    detection_id: str = None
    suppressed_duplicate: bool = False
    duplicate_of_detection_id: str = None
    cluster_index: int = None
    track_id: int = None
    camera_track_id: int = None
    camera_track_confirmed: bool = None
    global_reassociation_update_only: bool = False
    global_birth_pending: bool = False
    global_birth_private_track_id: int = None


def box_quality_ok(box, width, height):
    """Visible (image-clipped) detector box must be big enough and not too wide."""
    box = np.asarray(box, dtype=np.float64).reshape(4)
    if not np.isfinite(box).all():
        return False
    clipped = np.array([np.clip(box[0], 0.0, width), np.clip(box[1], 0.0, height), np.clip(box[2], 0.0, width), np.clip(box[3], 0.0, height)])
    visible_width, visible_height = max(0.0, float(clipped[2] - clipped[0])), max(0.0, float(clipped[3] - clipped[1]))
    return not (box[2] - box[0] <= 0.0 or box[3] - box[1] <= 0.0 or visible_width <= 0.0 or visible_height <= 0.0
                or visible_width < max(5.0, 0.004 * width) or visible_height < max(10.0, 0.010 * height)
                or (visible_height > 0.0 and visible_width / visible_height > 3.25))


def pose_quality_ok(keypoints, box, width, height):
    """MHR-70 anatomy gate: >= 3 core joints in frame and inside the (padded) detector box, >= 3 foot contacts in frame, and the
    feet near the box bottom."""
    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] < 21:
        return False
    box = np.asarray(box, dtype=np.float64).reshape(4)
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return False
    clipped = np.array([np.clip(box[0], 0.0, width), np.clip(box[1], 0.0, height), np.clip(box[2], 0.0, width), np.clip(box[3], 0.0, height)])
    box_height = max(0.0, float(clipped[3] - clipped[1]))
    if max(0.0, float(clipped[2] - clipped[0])) <= 0.0 or box_height <= 0.0:
        return False
    scale = max(box_height, 12.0)
    padding = max(6.0, 0.06 * scale)
    relaxed = np.array([clipped[0] - padding, clipped[1] - padding, clipped[2] + padding, clipped[3] + padding])
    xy = points[:, :2]
    present = np.isfinite(points).all(axis=1) & np.any(np.abs(xy) > 1.0e-5, axis=1)
    if points.shape[1] >= 3:
        present &= points[:, 2] >= 0.05
    in_frame = present & (xy[:, 0] >= 0.0) & (xy[:, 0] < width) & (xy[:, 1] >= 0.0) & (xy[:, 1] < height)
    in_relaxed = in_frame & (xy[:, 0] >= relaxed[0]) & (xy[:, 0] <= relaxed[2]) & (xy[:, 1] >= relaxed[1]) & (xy[:, 1] <= relaxed[3])
    core, contacts = np.array([5, 6, 9, 10, 11, 12]), np.arange(13, 21)
    visible_contacts = xy[contacts[in_frame[contacts]]]
    if len(visible_contacts) < 3 or int(in_relaxed[core].sum()) < 3:  # in_relaxed implies in_frame
        return False
    gap = float(clipped[3] - float(np.median(visible_contacts[:, 1])))
    return max(0.0, gap) / scale <= 0.55 and max(0.0, -gap) / scale <= 0.50


def detector_box(raw):
    """SAM3's own box (before the 1.2x expansion for the body model), from the cache's detector metadata."""
    return np.asarray(raw.model_fields["detector_metadata"]["raw_bbox"], dtype=np.float64).reshape(-1)


def foot_pixel(keypoints, width, height):
    """Median toe/heel contact of each foot, averaged (None unless both feet have a visible contact)."""
    keypoints = np.asarray(keypoints, dtype=np.float64)

    def valid(indices):
        points = []
        for index in indices:
            point = keypoints[index].reshape(-1)
            if not np.isfinite(point[:2]).all() or (point.size >= 3 and (not np.isfinite(point[2]) or point[2] < 0.0)):
                continue
            if (point[:2] <= 0.0).any() or point[0] >= width or point[1] >= height:
                continue
            points.append(point[:2])
        return points

    left, right = valid((15, 16, 17)), valid((18, 19, 20))
    if left and right:
        return (np.median(np.stack(left, axis=0), axis=0) + np.median(np.stack(right, axis=0), axis=0)) / 2.0
    return None


def project_detections(raw_detections, camera, timestamp_ns):
    """Pose-gate each SAM detection (its box already passed box_quality_ok in pass 1), pick its foot pixel, and intersect it
    with the floor (z=0)."""
    width, height = camera["width"], camera["height"]
    projected = []
    for index, raw in enumerate(raw_detections):
        if not pose_quality_ok(raw.keypoints_2d, detector_box(raw), width, height):
            continue
        foot = foot_pixel(raw.keypoints_2d, width, height)
        if foot is None:
            continue
        x0, y0, x1, y1 = int(np.floor(foot[0])), int(np.floor(foot[1])), int(np.ceil(foot[0])), int(np.ceil(foot[1]))
        if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height or not camera["valid"][y0:y1 + 1, x0:x1 + 1].all():
            continue
        ground_xy = pixel_to_ground(camera, foot)
        if ground_xy is None or not np.isfinite(ground_xy).all() or float(np.linalg.norm(ground_xy)) > 50.0:
            continue
        projected.append(ProjectedDetection(
            camera_index=camera["index"], timestamp_ns=timestamp_ns, raw=raw,
            foot_pixel_uv=np.asarray(foot, dtype=np.float64), ground_xy=np.asarray(ground_xy, dtype=np.float64),
            ground_yaw=sam_yaw_to_ground(raw.global_rot_zyx, camera),
            detection_id=f"{timestamp_ns}:cam{camera['index']}:det{index}"))
    return projected
