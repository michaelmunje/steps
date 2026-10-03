from collections import deque

import cv2
import numpy as np

COLORS = ((55, 55, 235), (0, 155, 255), (0, 215, 235), (70, 195, 70), (225, 115, 25), (205, 75, 195))
FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA


def xy(value):
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array[:2].copy() if array.size >= 2 and np.isfinite(array[:2]).all() else None


def darken(color, factor=0.55):
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)


def text_box(image, text, origin, color=(255, 255, 255), background=(25, 25, 25), scale=0.48, thickness=1, padding=4):
    if not text:
        return
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x = int(np.clip(origin[0], 0, max(0, image.shape[1] - 1)))
    y = int(np.clip(origin[1], height + padding, max(height + padding, image.shape[0] - 1)))
    cv2.rectangle(image, (max(0, x - padding), max(0, y - height - padding)),
                  (min(image.shape[1] - 1, x + width + padding), min(image.shape[0] - 1, y + baseline + padding)), background, -1)
    cv2.putText(image, text, (x, y), FONT, scale, color, thickness, AA)


def letterbox(image, size=(480, 360), background=(32, 35, 39)):
    scale = min(size[0] / image.shape[1], size[1] / image.shape[0])
    width, height = max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    offset = ((size[0] - width) // 2, (size[1] - height) // 2)
    canvas = np.full((size[1], size[0], 3), background, dtype=np.uint8)
    canvas[offset[1]:offset[1] + height, offset[0]:offset[0] + width] = resized
    return canvas, float(scale), offset


def map_point(point, scale, offset):
    point = xy(point)
    return None if point is None else (int(round(offset[0] + scale * point[0])), int(round(offset[1] + scale * point[1])))


def exact_mean(person):
    positions = [p for p in (xy(m.ground_xy) for m in person.members) if p is not None]
    if positions:
        return np.mean(np.stack(positions, axis=0), axis=0)
    for value in (person.raw_ground_xy, person.ground_xy):
        if xy(value) is not None:
            return xy(value)
    return None


def yaw_concentration(person):
    if person.yaw_concentration is not None and np.isfinite(float(person.yaw_concentration)):
        return float(np.clip(float(person.yaw_concentration), 0.0, 1.0))
    yaws = [float(m.ground_yaw) for m in person.members if m.ground_yaw is not None and np.isfinite(float(m.ground_yaw))]
    if not yaws:
        return None
    return float(np.clip(np.linalg.norm(np.array([np.cos(yaws).mean(), np.sin(yaws).mean()])), 0.0, 1.0))


def person_label(person, index):
    if person.track_id is not None:
        return f"T{person.track_id}"
    clusters = {m.cluster_index for m in person.members if m.cluster_index is not None}
    return f"C{next(iter(clusters))}" if len(clusters) == 1 else f"P{index}"


def retrospective_without_tracks(fused, tracked):
    tracked_ids, emitted, selected = {int(t.track_id) for t in tracked}, set(), []
    for index, person in enumerate(fused):
        if person.global_birth_retroactively_labeled and person.track_id is not None and int(person.track_id) not in tracked_ids | emitted:
            emitted.add(int(person.track_id))
            selected.append((index, person))
    return selected


def cameras_text(indices):
    cameras = sorted({int(c) for c in indices})
    return ",".join(str(c) for c in cameras) if cameras else "-"


def yaw_text(yaw):
    try:
        yaw = float(yaw)
    except (TypeError, ValueError):
        return "n/a"
    return f"{np.degrees((yaw + np.pi) % (2.0 * np.pi) - np.pi):+.0f}deg" if np.isfinite(yaw) else "n/a"


class Renderer:
    """Draws the camera mosaic, the diagnostic BEV + track panel, and the clean combined view."""

    def __init__(self, cameras, extent=(-4.0, 14.0, -3.0, 15.0), bev_size=720, trajectory_length=60):
        self.cameras = cameras
        self.extent = extent
        self.bev_size = bev_size
        self.trajectory_length = trajectory_length
        drawable = float(bev_size - 1)
        self.ppm = drawable / max(extent[1] - extent[0], extent[3] - extent[2])
        width, height = (extent[1] - extent[0]) * self.ppm, (extent[3] - extent[2]) * self.ppm
        left, top = (drawable - width) / 2.0, (drawable - height) / 2.0
        self.viewport = (left, top, left + width, top + height)
        self.last_frames = {}
        self.trajectories = {}
        self.trajectory_stamp = {}

    def to_canvas(self, ground):
        x, y = (float(v) for v in np.asarray(tuple(ground)).reshape(2))
        return self.viewport[0] + (x - self.extent[0]) * self.ppm, self.viewport[1] + (self.extent[3] - y) * self.ppm

    def to_pixel(self, ground):
        px, py = self.to_canvas(ground)
        return int(round(px)), int(round(py))

    def viewport_pixels(self):
        return tuple(int(round(v)) for v in self.viewport)

    def visible(self, ground):
        px, py = self.to_canvas(ground)
        return self.viewport[0] <= px <= self.viewport[2] and self.viewport[1] <= py <= self.viewport[3]

    def render(self, frame):
        """Return (camera mosaic, standalone diagnostic BEV + info panel, mosaic|clean BEV)."""
        mosaic = self.camera_mosaic(frame)
        bev = np.ascontiguousarray(np.hstack((self.diagnostic_bev(frame), np.full((self.bev_size, 4, 3), (20, 22, 25), dtype=np.uint8),
                                              self.information_panel(frame))))
        clean = self.clean_bev(frame)
        if clean.shape[0] != mosaic.shape[0]:
            clean = cv2.resize(clean, (max(1, int(round(clean.shape[1] * mosaic.shape[0] / clean.shape[0]))), mosaic.shape[0]), interpolation=cv2.INTER_AREA)
        combined = np.ascontiguousarray(np.hstack((mosaic, np.full((mosaic.shape[0], 4, 3), (30, 30, 30), dtype=np.uint8), clean)))
        return mosaic, bev, combined

    def draw_raw(self, tile, raw, scale, offset, color, bbox_thickness=2, keypoint_radius=1):
        bbox = np.asarray(raw.bbox_xyxy, dtype=np.float64).reshape(-1)
        top_left = None
        if bbox.size >= 4 and np.isfinite(bbox[:4]).all():
            top_left, bottom_right = map_point(bbox[:2], scale, offset), map_point(bbox[2:4], scale, offset)
            if top_left is not None and bottom_right is not None:
                cv2.rectangle(tile, top_left, bottom_right, color, bbox_thickness, AA)
        keypoints = np.asarray(raw.keypoints_2d, dtype=np.float64)
        if keypoints.ndim == 2 and keypoints.shape[1] >= 2:
            for point in keypoints[:, :2]:
                if not np.isfinite(point).all() or not (point > 0.0).all():
                    continue
                mapped = map_point(point, scale, offset)
                if mapped is not None and 0 <= mapped[0] < tile.shape[1] and 0 <= mapped[1] < tile.shape[0]:
                    cv2.circle(tile, mapped, keypoint_radius, darken(color, 0.9), -1, AA)
        return top_left

    def camera_mosaic(self, frame):
        person_by_member = {id(m): (p, i) for i, p in enumerate(frame["fused"]) for m in p.members}
        tiles = []
        for camera in range(6):
            color = COLORS[camera]
            detections = [d for d in frame["projected"] if d.camera_index == camera]
            if camera not in frame["images"]:
                held = self.last_frames.get(camera)
                if held is None:
                    tile = np.full((360, 480, 3), (42, 45, 50), dtype=np.uint8)
                    cv2.rectangle(tile, (0, 0), (479, 359), color, 3)
                    cv2.line(tile, (20, 20), (460, 340), (70, 74, 80), 2)
                    cv2.line(tile, (460, 20), (20, 340), (70, 74, 80), 2)
                    (tw, th), _ = cv2.getTextSize(f"cam{camera}: NO FRAME", FONT, 0.8, 2)
                    cv2.putText(tile, f"cam{camera}: NO FRAME", ((480 - tw) // 2, (360 + th) // 2), FONT, 0.8, (210, 210, 210), 2, AA)
                else:
                    tile = letterbox(held[0])[0]
                    tile = np.clip(tile.astype(np.float32) * 0.32, 0, 255).astype(np.uint8)
                    for start in range(-tile.shape[0], tile.shape[1], 42):
                        cv2.line(tile, (start, 0), (start + tile.shape[0], tile.shape[0] - 1), (72, 72, 72), 1, AA)
                    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), color, 3)
                    text_box(tile, f"cam{camera}  STALE +{max(0.0, (frame['timestamp_ns'] - held[1]) / 1e9):.2f}s", (8, 24),
                             color=(235, 235, 235), background=(35, 35, 120), scale=0.58, thickness=2)
                tiles.append(tile)
                continue

            stamp, image = frame["images"][camera]
            tile, scale, offset = letterbox(image)
            self.last_frames[camera] = (image, int(stamp))
            for index, detection in enumerate(detections):
                top_left = self.draw_raw(tile, detection.raw, scale, offset, color)
                foot = map_point(detection.foot_pixel_uv, scale, offset)
                if foot is not None:
                    cv2.circle(tile, foot, 6, (0, 255, 255), -1, AA)
                    cv2.circle(tile, foot, 6, (15, 15, 15), 1, AA)
                mean = None
                if detection.suppressed_duplicate:
                    identity = f"DUP->{str(detection.duplicate_of_detection_id).rsplit(':', 1)[-1]}" if detection.duplicate_of_detection_id is not None else "DUP"
                elif id(detection) in person_by_member:
                    person, person_index = person_by_member[id(detection)]
                    identity, mean = person_label(person, person_index), exact_mean(person)
                else:
                    identity = (f"T{detection.track_id}" if detection.track_id is not None else
                                f"C{detection.cluster_index}" if detection.cluster_index is not None else f"D{index}")
                parts = [identity]
                raw_xy = xy(detection.ground_xy)
                if raw_xy is not None:
                    parts.append(f"raw({raw_xy[0]:+.2f},{raw_xy[1]:+.2f})")
                if mean is not None:
                    parts.append(f"mean({mean[0]:+.2f},{mean[1]:+.2f})")
                if detection.ground_yaw is not None and np.isfinite(float(detection.ground_yaw)):
                    parts.append(f"yaw {np.degrees(float(detection.ground_yaw)):+.0f}deg")
                origin = (max(5, top_left[0] if top_left is not None else (foot[0] if foot else 5)),
                          max(22, (top_left[1] - 5) if top_left is not None else (foot[1] if foot else 24)))
                text_box(tile, "  ".join(parts), origin, background=darken(color, 0.34), scale=0.40)
            projected_raw = {id(d.raw) for d in detections}
            no_ground, seen = [], set(projected_raw)
            for raw in frame["raw"].get(camera, ()):
                if id(raw) not in seen:
                    seen.add(id(raw))
                    no_ground.append(raw)
            for index, raw in enumerate(no_ground):
                top_left = self.draw_raw(tile, raw, scale, offset, (35, 35, 245), bbox_thickness=2, keypoint_radius=2)
                confidence = float(raw.confidence) if raw.confidence is not None else None
                confidence_text = f"  conf={confidence:.2f}" if confidence is not None and np.isfinite(confidence) else ""
                text_box(tile, f"D{index}  NO GROUND{confidence_text}",
                         (max(5, top_left[0] if top_left is not None else 5), max(48, (top_left[1] - 5) if top_left is not None else 48)),
                         background=(30, 30, 170), scale=0.43)
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), color, 3)
            duplicates = sum(bool(d.suppressed_duplicate) for d in detections)
            summary = f"{len(detections) - duplicates} ground"
            if duplicates:
                summary += f"  {duplicates} DUP"
            if no_ground:
                summary += f"  {len(no_ground)} NO GROUND"
            text_box(tile, f"cam{camera}  {summary}  dt={(stamp - frame['timestamp_ns']) / 1e6:+.1f}ms", (8, 24),
                     background=darken(color, 0.30), scale=0.54, thickness=2)
            tiles.append(tile)
        return np.ascontiguousarray(np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])]))

    def background(self, axis_title):
        panel = np.full((self.bev_size, self.bev_size, 3), (218, 221, 225), dtype=np.uint8)
        left, top, right, bottom = self.viewport_pixels()
        cv2.rectangle(panel, (left, top), (right, bottom), (242, 240, 234), -1)
        x_min, x_max, y_min, y_max = self.extent
        for x in range(int(np.ceil(x_min)), int(np.floor(x_max)) + 1):
            px, _ = self.to_pixel((x, y_min))
            cv2.line(panel, (px, top), (px, bottom), (170, 175, 180) if x == 0 else (215, 213, 207), 2 if x == 0 else 1, AA)
            if x % 2 == 0:
                cv2.putText(panel, str(x), (px + 3, max(top + 13, bottom - 8)), FONT, 0.38, (105, 105, 105), 1, AA)
        for y in range(int(np.ceil(y_min)), int(np.floor(y_max)) + 1):
            _, py = self.to_pixel((x_min, y))
            cv2.line(panel, (left, py), (right, py), (170, 175, 180) if y == 0 else (215, 213, 207), 2 if y == 0 else 1, AA)
            if y % 2 == 0:
                cv2.putText(panel, str(y), (left + 5, max(top + 13, py - 3)), FONT, 0.38, (105, 105, 105), 1, AA)
        cv2.rectangle(panel, (left, top), (right, bottom), (75, 78, 82), 2)
        cv2.rectangle(panel, (0, 0), (self.bev_size - 1, self.bev_size - 1), (120, 124, 130), 1)
        if axis_title:
            text_box(panel, "Ground frame: +X right, +Y up", (left + 10, top + 23), color=(50, 50, 50), background=(242, 240, 234))
        for index, camera in sorted(self.cameras.items()):
            position = xy(camera["t"])
            if position is None or not self.visible(position):
                continue
            center, color = self.to_pixel(position), COLORS[index]
            facing = camera["R"] @ np.array([0.0, 0.0, 1.0])
            if float(np.linalg.norm(facing[:2])) > 1e-9:
                yaw = float(np.arctan2(facing[1], facing[0]))
                if np.isfinite(yaw):
                    cv2.arrowedLine(panel, center, self.to_pixel(position + 1.0 * np.array([np.cos(yaw), np.sin(yaw)])), color, 2, AA, tipLength=0.28)
            cv2.circle(panel, center, 6, darken(color, 0.55), -1, AA)
            cv2.circle(panel, center, 4, color, -1, AA)
            cv2.putText(panel, f"c{index}", (center[0] + 7, center[1] - 5), FONT, 0.32, darken(color, 0.65), 1, AA)
        return panel

    def arrow(self, panel, position, yaw, concentration, observed, compact):
        position = xy(position)
        if position is None or not self.visible(position):
            return
        try:
            yaw = float(yaw)
        except (TypeError, ValueError):
            return
        if not np.isfinite(yaw):
            return
        concentration = None if concentration is None else float(concentration)
        if concentration is not None and (not np.isfinite(concentration) or concentration < 0.20):
            return
        reliable = concentration is None or (np.isfinite(concentration) and concentration >= 0.35)
        length = max(1, int(round(0.70 * self.ppm)))
        origin = self.to_pixel(position)
        end = (int(round(origin[0] + np.cos(yaw) * length)), int(round(origin[1] - np.sin(yaw) * length)))
        color = (130, 130, 150) if not observed else (35, 45, 195) if reliable else (95, 110, 165)
        cv2.arrowedLine(panel, origin, end, color, 2 if compact else 3, AA, tipLength=0.28)

    def clean_person(self, panel, person, index):
        position = xy(person.smoothed_xy)
        position = exact_mean(person) if position is None else position
        if position is None or not self.visible(position):
            return
        point = self.to_pixel(position)
        self.arrow(panel, position, person.ground_yaw, yaw_concentration(person), True, True)
        cv2.circle(panel, point, 6, (35, 35, 35), -1, AA)
        cv2.circle(panel, point, 4, (245, 245, 245), -1, AA)
        text_box(panel, person_label(person, index), (point[0] + 8, point[1] - 7), color=(245, 245, 245), background=(42, 42, 42), scale=0.30, padding=2)

    def tracked_person(self, panel, track):
        observed = bool(track.observed_this_frame)
        position = xy(track.position_ground_xy) if observed else xy(track.last_observed_position_ground_xy)
        if position is None or not self.visible(position):
            return
        point = self.to_pixel(position)
        self.arrow(panel, position, track.ground_yaw, track.yaw_concentration, observed, True)
        cv2.circle(panel, point, 6, (35, 35, 35) if observed else (85, 85, 95), -1, AA)
        cv2.circle(panel, point, 4, (245, 245, 245) if observed else (185, 185, 195), -1, AA)
        text_box(panel, f"T{track.track_id}" if observed else f"T{track.track_id} STALE", (point[0] + 8, point[1] - 7),
                 color=(245, 245, 245), background=(42, 42, 42), scale=0.30, padding=2)

    def clean_bev(self, frame):
        panel = self.background(axis_title=False)
        for track in frame["tracked"]:
            self.tracked_person(panel, track)
        for index, person in retrospective_without_tracks(frame["fused"], frame["tracked"]):
            self.clean_person(panel, person, index)
        return np.ascontiguousarray(panel)

    def diagnostic_bev(self, frame):
        panel = self.background(axis_title=True)
        fused, detections = frame["fused"], frame["projected"]
        for person in fused:
            if person.track_id is None or self.trajectory_stamp.get(int(person.track_id)) == frame["timestamp_ns"]:
                continue
            position = xy(person.smoothed_xy)
            position = exact_mean(person) if position is None else position
            if position is None:
                continue
            self.trajectories.setdefault(int(person.track_id), deque(maxlen=self.trajectory_length)).append(position)
            self.trajectory_stamp[int(person.track_id)] = frame["timestamp_ns"]

        for index, person in enumerate(fused):
            mean = exact_mean(person)
            if mean is None:
                continue
            mean_visible, mean_pixel = self.visible(mean), self.to_pixel(mean)
            if mean_visible:
                for member in person.members:
                    member_xy = xy(member.ground_xy)
                    if member_xy is not None and self.visible(member_xy):
                        cv2.line(panel, self.to_pixel(member_xy), mean_pixel, darken(COLORS[member.camera_index], 0.72), 1, AA)
            if person.track_id is not None:
                history = self.trajectories.get(int(person.track_id))
                points = [self.to_pixel(p) for p in history if self.visible(p)] if history is not None and len(history) >= 2 else []
                for k in range(1, len(points)):
                    fraction = k / max(1, len(points) - 1)
                    shade = int(120 + 95 * fraction)
                    cv2.line(panel, points[k - 1], points[k], (shade, shade, shade), 1 if fraction < 0.6 else 2, AA)
            for member in person.members:
                member_xy = xy(member.ground_xy)
                if member_xy is None or not self.visible(member_xy):
                    continue
                point, color = self.to_pixel(member_xy), COLORS[member.camera_index]
                cv2.circle(panel, point, 6, color, -1, AA)
                cv2.circle(panel, point, 6, (35, 35, 35), 1, AA)
                cv2.putText(panel, str(member.camera_index), (point[0] + 8, point[1] - 6), FONT, 0.34, darken(color, 0.62), 1, AA)
            if not mean_visible:
                continue
            cv2.circle(panel, mean_pixel, 13, (25, 25, 25), -1, AA)
            cv2.circle(panel, mean_pixel, 10, (250, 250, 250), -1, AA)
            cv2.circle(panel, mean_pixel, 10, (45, 45, 45), 2, AA)
            concentration = yaw_concentration(person)
            self.arrow(panel, mean, person.ground_yaw, concentration, True, False)
            smoothed = xy(person.smoothed_xy)
            if smoothed is not None and np.linalg.norm(smoothed - mean) > 1e-3 and self.visible(smoothed):
                smooth_pixel = self.to_pixel(smoothed)
                cv2.drawMarker(panel, smooth_pixel, (35, 35, 35), cv2.MARKER_DIAMOND, 13, 2, AA)
                cv2.line(panel, mean_pixel, smooth_pixel, (100, 100, 100), 1, AA)
            spread = f" spread={float(person.spread_m):.2f}m" if person.spread_m is not None else ""
            yaw_q = f" yawQ={concentration:.2f}" if concentration is not None else ""
            text_box(panel, f"{person_label(person, index)} mean=({mean[0]:+.2f},{mean[1]:+.2f}) cams={cameras_text(person.camera_indices)}"
                     f" n={len(person.members)}{spread}{yaw_q}", (mean_pixel[0] + 15, mean_pixel[1] - 14),
                     color=(245, 245, 245), background=(42, 42, 42), scale=0.42)

        for track in frame["tracked"]:
            if not track.observed_this_frame:
                self.tracked_person(panel, track)
        members = {id(m) for p in fused for m in p.members}
        for detection in detections:
            ground = xy(detection.ground_xy)
            if id(detection) in members or detection.suppressed_duplicate or ground is None or not self.visible(ground):
                continue
            cv2.drawMarker(panel, self.to_pixel(ground), COLORS[detection.camera_index], cv2.MARKER_TILTED_CROSS, 14, 2, AA)

        stamps = [stamp for stamp, _ in frame["images"].values()]
        skew = max(stamps) - min(stamps) if len(stamps) >= 2 else 0
        eligible = lambda d: not d.suppressed_duplicate and (d.camera_track_confirmed is not False or d.global_reassociation_update_only)
        unconfirmed = sum(not d.suppressed_duplicate and d.camera_track_confirmed is False and not d.global_reassociation_update_only for d in detections)
        left, top, _, _ = self.viewport_pixels()
        text_box(panel, f"t={frame['timestamp_ns'] / 1e9:.3f}s  cams={sorted(frame['images'])}  skew={skew / 1e6:.1f}ms  "
                 f"projected={len(detections)}  assoc={sum(eligible(d) for d in detections)}  unconfirmed={unconfirmed}  fused={len(fused)}",
                 (left + 10, top + 47), color=(245, 245, 245), background=(62, 65, 70), scale=0.43)
        for camera in range(6):
            x = 12 + camera * 65
            cv2.circle(panel, (x, self.bev_size - 29), 5, COLORS[camera], -1, AA)
            cv2.putText(panel, f"cam{camera}", (x + 8, self.bev_size - 25), FONT, 0.34, (60, 60, 60), 1, AA)
        return np.ascontiguousarray(panel)

    def information_panel(self, frame):
        width = max(300, int(round(self.bev_size * 0.50)))
        panel = np.full((self.bev_size, width, 3), (37, 40, 44), dtype=np.uint8)
        text_box(panel, "TRACK DETAILS", (12, 25), color=(248, 248, 248), background=(37, 40, 44), scale=0.50, padding=1)
        text_box(panel, f"t={frame['timestamp_ns'] / 1e9:.3f}s  fresh={cameras_text(frame['images'])}  "
                 f"projected={len(frame['projected'])}  fused={len(frame['fused'])}", (12, 48),
                 color=(205, 210, 216), background=(37, 40, 44), scale=0.31, padding=1)
        for camera in range(6):
            cv2.circle(panel, (13 + camera * 47, 70), 4, COLORS[camera], -1, AA)
            cv2.putText(panel, f"c{camera}", (19 + camera * 47, 74), FONT, 0.28, (205, 210, 216), 1, AA)
        cards = []
        for track in sorted(frame["tracked"], key=lambda t: t.track_id):
            observed = bool(track.observed_this_frame)
            position = xy(track.position_ground_xy) if observed else xy(track.last_observed_position_ground_xy)
            if position is None:
                continue
            velocity = np.asarray(track.velocity_ground_xy_mps, dtype=np.float64)
            if not observed:
                state = f"missed={track.missed_seconds:.2f}s"
            elif track.position_covariance is not None and velocity.shape == (2,) and np.all(np.isfinite(velocity)):
                state = f"filtered_speed={np.linalg.norm(velocity):.2f}m/s  age={track.age_steps}f"
            else:
                state = f"age={track.age_steps}f"
            cards.append((f"T{track.track_id}  {'OBSERVED' if observed else 'STALE'}",
                          f"xy=({position[0]:+.2f},{position[1]:+.2f})m  yaw={yaw_text(track.ground_yaw)}",
                          f"{'cams' if observed else 'lastcams'}={cameras_text(track.camera_indices)}  spread={track.spread_m:.2f}m  {state}  hits={track.hit_count}"))
        top0 = 91
        card_height = 52 if not cards else min(52, max(30, max(1, self.bev_size - top0 - 8) // len(cards)))
        spacing = min(15, max(9, (card_height - 5) // 3))
        compact = card_height < 48
        divider = min(8, max(4, card_height // 6))
        for index, lines in enumerate(cards):
            top = top0 + index * card_height
            cv2.line(panel, (10, top - divider), (width - 10, top - divider), (73, 77, 82), 1, AA)
            for k, line in enumerate(lines):
                text_box(panel, line, (12, top + k * spacing), color=(245, 245, 245) if k == 0 else (205, 210, 216), background=(37, 40, 44),
                         scale=((0.30 if compact else 0.34) if k == 0 else (0.25 if compact else 0.29)), padding=1)
        if not cards:
            text_box(panel, "No active observed tracks", (12, top0), color=(175, 180, 186), background=(37, 40, 44), scale=0.36, padding=1)
        return panel
