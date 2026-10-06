from dataclasses import dataclass

import numpy as np


def measurement_noise(std_m):
    """(covariance R, log det R, metres per unit of Mahalanobis distance) for a camera foot point with std_m noise per axis."""
    R = np.eye(2) * std_m ** 2
    _, logdet = np.linalg.slogdet(R)
    return R, logdet, float(np.exp(0.25 * logdet))


@dataclass
class FusedPerson:
    members: list
    ground_xy: np.ndarray
    ground_yaw: float
    spread_m: float
    track_id: int = None
    smoothed_xy: np.ndarray = None
    yaw_concentration: float = None
    raw_ground_xy: np.ndarray = None
    track_prediction_guided: bool = False
    global_track_bootstrap_only: bool = False
    global_birth_pending: bool = False
    global_birth_private_track_id: int = None
    global_birth_consecutive_started_timestamp_ns: int = None
    global_birth_confirmed: bool = False
    global_birth_retroactively_labeled: bool = False

    @property
    def camera_indices(self):
        return sorted(member.camera_index for member in self.members)


@dataclass
class TrackedPerson:
    track_id: int
    position_ground_xy: np.ndarray
    velocity_ground_xy_mps: np.ndarray
    ground_yaw: float
    spread_m: float
    yaw_concentration: float
    observed_this_frame: bool
    missed_steps: int
    missed_seconds: float
    age_steps: int
    hit_count: int
    confirmed: bool
    last_observed_timestamp_ns: int
    camera_indices: tuple = ()
    position_covariance: np.ndarray = None
    consecutive_hit_count: int = 0
    consecutive_observed_seconds: float = 0.0
    last_observed_position_ground_xy: np.ndarray = None


@dataclass(frozen=True)
class GlobalBirthPromotion:
    private_track_id: int
    public_track_id: int
    consecutive_started_timestamp_ns: int
    promotion_timestamp_ns: int


def linear_sum_assignment(costs):
    """Optimal one-to-one assignment (SciPy); rows sorted."""
    from scipy.optimize import linear_sum_assignment as scipy_assignment
    rows, columns = scipy_assignment(np.asarray(costs, dtype=np.float64))
    return rows.astype(np.int64), columns.astype(np.int64)


def wrap(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def axial_yaw_difference(left, right):
    directed = abs(float(np.arctan2(np.sin(float(left) - float(right)), np.cos(float(left) - float(right)))))
    return float(min(directed, np.pi - directed))


def axial_branch_nearest(yaw, reference, tie_reference=None):
    """Return yaw or yaw+pi, whichever points toward the reference."""
    yaw, reference = wrap(float(yaw)), wrap(float(reference))
    alignment = float(np.cos(yaw - reference))
    if alignment > 1e-9:
        return yaw
    opposite = wrap(yaw + np.pi)
    if alignment < -1e-9:
        return opposite
    if tie_reference is None:
        return yaw
    return min((yaw, opposite), key=lambda candidate: abs(wrap(float(candidate) - float(tie_reference))))


def suppress_same_camera_duplicates(observations, max_distance_m):
    """Per camera, greedily keep detections by confidence; mark any whose ground foot lies within max_distance_m of a kept one as a duplicate.
    Returns the kept detections."""
    for observation in observations:
        observation.suppressed_duplicate = False
        observation.duplicate_of_detection_id = None
    by_camera = {}
    for index, observation in enumerate(observations):
        by_camera.setdefault(observation.camera_index, []).append(index)

    kept = set()
    for camera in sorted(by_camera):
        winners = []
        for index in sorted(by_camera[camera], key=lambda i: (-float(observations[i].raw.confidence), i)):
            near = [w for w in winners if np.linalg.norm(observations[index].ground_xy - observations[w].ground_xy) <= max_distance_m]
            if near:
                observations[index].suppressed_duplicate, observations[index].duplicate_of_detection_id = True, observations[near[0]].detection_id
            else:
                winners.append(index)
                kept.add(index)
    return [o for i, o in enumerate(observations) if i in kept]


def fuse_orientations(members):
    """Axial (front/back-flip tolerant) mean of member yaws; the branch (front vs back) nearest the first member's yaw."""
    values = np.asarray([float(m.ground_yaw) for m in members if m.ground_yaw is not None and np.isfinite(m.ground_yaw)], dtype=np.float64)
    if values.size == 0:
        return None, None
    vector = np.sum(np.exp(2j * values)) / float(values.size)
    if abs(vector) < 1e-12:
        return None, 0.0
    axis = 0.5 * float(np.angle(vector))
    concentration = float(np.clip(abs(vector), 0.0, 1.0))
    if np.cos(axis - float(values[0])) < 0.0:
        axis += np.pi
    return wrap(axis), concentration


def complete_link_clusters(observations, max_distance_m, yaw_weight):
    """Hungarian-match every camera pair (within max_distance_m), then merge matched pairs, cheapest first, into multiview cliques:
    two clusters merge only if they share no camera and every cross pair between them was matched."""
    if not observations:
        return []
    by_camera = {}
    for index, observation in enumerate(observations):
        by_camera.setdefault(observation.camera_index, []).append(index)
    edges = []
    cameras = sorted(by_camera)
    for a, left_camera in enumerate(cameras):
        for right_camera in cameras[a + 1:]:
            left, right = by_camera[left_camera], by_camera[right_camera]
            size = len(left) + len(right)
            invalid = (max_distance_m + yaw_weight * (np.pi / 2.0) + max_distance_m + 1.0) * (size + 1)
            costs = np.full((size, size), invalid)
            costs[:len(left), len(right):] = max_distance_m / 2.0
            costs[len(left):, :len(right)] = max_distance_m / 2.0
            costs[len(left):, len(right):] = 0.0
            left_xy = np.array([observations[i].ground_xy for i in left], dtype=np.float64)
            right_xy = np.array([observations[j].ground_xy for j in right], dtype=np.float64)
            distances = np.sqrt(((left_xy[:, None, :] - right_xy[None, :, :]) ** 2).sum(axis=2))
            valid = distances <= max_distance_m
            costs[:len(left), :len(right)][valid] = distances[valid]
            for row, column in zip(*np.nonzero(valid)):
                yi, yj = observations[left[row]].ground_yaw, observations[right[column]].ground_yaw
                if yi is not None and yj is not None and np.isfinite(yi) and np.isfinite(yj):
                    costs[row, column] += yaw_weight * axial_yaw_difference(float(yi), float(yj))
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows.tolist(), columns.tolist()):
                if row >= len(left) or column >= len(right) or not valid[row, column]:
                    continue
                i, j = left[row], right[column]
                edges.append((float(costs[row, column]), min(i, j), max(i, j)))

    accepted = {(i, j) for _, i, j in edges}
    parent = list(range(len(observations)))
    members = {i: [i] for i in range(len(observations))}
    camera_sets = {i: {o.camera_index} for i, o in enumerate(observations)}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges.sort()
    for _, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj or camera_sets[ri] & camera_sets[rj]:
            continue
        if not all((min(a, b), max(a, b)) in accepted for a in members[ri] for b in members[rj]):
            continue
        keep, drop = min(ri, rj), max(ri, rj)
        parent[drop] = keep
        members[keep] = sorted(members[keep] + members[drop])
        camera_sets[keep] = camera_sets[keep] | camera_sets[drop]
        del members[drop], camera_sets[drop]
    return sorted(members.values(), key=lambda cluster: cluster[0])


class Tracker:
    """Global ground-plane [x, y, vx, vy] constant-velocity Kalman tracks with axial yaw smoothing. Measurements arrive already
    assigned to a track (FusedPerson.track_id from associate_and_fuse) or as confirmed births, which start new tracks."""

    def __init__(self, R, max_missing_seconds, acceleration_noise, max_speed, yaw_gain):
        self.R = R
        self.max_missing_seconds = max_missing_seconds
        self.acceleration_noise = acceleration_noise
        self.max_speed = max_speed
        self.yaw_gain = yaw_gain
        self.tracks = {}
        self.next_track_id = 0
        self.last_prediction_ns = None
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def consecutive_seconds(self, track):
        if track["consecutive_started_ns"] is None or track["consecutive_hits"] <= 0:
            return 0.0
        return max(0.0, (track["last_observed_ns"] - track["consecutive_started_ns"]) / 1e9)

    def within_coast(self, track, timestamp_ns):
        return (timestamp_ns - track["last_observed_ns"]) / 1e9 <= self.max_missing_seconds

    def update_yaw(self, track, detection):
        if detection.ground_yaw is None or not np.isfinite(detection.ground_yaw):
            return
        concentration = detection.yaw_concentration
        concentration = 1.0 if concentration is None or not np.isfinite(concentration) else concentration
        concentration = float(np.clip(concentration, 0.0, 1.0))
        observed = wrap(float(detection.ground_yaw))
        if track["yaw"] is None:
            track["yaw"] = observed
        else:
            prior = track["yaw"]
            aligned = axial_branch_nearest(observed, prior, prior)
            track["yaw"] = wrap(prior + self.yaw_gain * wrap(float(aligned) - float(prior)))
        track["yaw_concentration"] = concentration if track["yaw_concentration"] is None else (
            (1.0 - self.yaw_gain) * track["yaw_concentration"] + self.yaw_gain * concentration)

    def update_metadata(self, track, detection):
        self.update_yaw(track, detection)
        spread = max(0.0, float(detection.spread_m))
        track["spread"] = spread if track["hit_count"] <= 1 else 0.75 * track["spread"] + 0.25 * spread
        track["camera_indices"] = tuple(detection.camera_indices)

    def predict(self, timestamp_ns):
        """Advance every track's Kalman state to timestamp_ns and drop tracks past the coast window."""
        timestamp_ns = int(timestamp_ns)
        if self.last_prediction_ns == timestamp_ns:
            return self.estimates()
        missed_before = {track_id: track["missed"] for track_id, track in self.tracks.items()}
        for track in self.tracks.values():
            dt = max(0.0, (timestamp_ns - track["last_ns"]) / 1e9)
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            Q = self.acceleration_noise ** 2 * np.array([[dt ** 3 / 3.0, 0.0, dt ** 2 / 2.0, 0.0], [0.0, dt ** 3 / 3.0, 0.0, dt ** 2 / 2.0],
                                                         [dt ** 2 / 2.0, 0.0, dt, 0.0], [0.0, dt ** 2 / 2.0, 0.0, dt]])
            Q += np.eye(4) * 1e-9
            track["state"] = F @ track["state"]
            track["covariance"] = F @ track["covariance"] @ F.T + Q
            track["last_ns"] = timestamp_ns
            track["missed"] += 1
            track["age"] += 1
            track["observed"] = False
        self.tracks = {track_id: track for track_id, track in self.tracks.items()
                       if missed_before[track_id] == 0 or self.within_coast(track, timestamp_ns)}
        self.last_prediction_ns = timestamp_ns
        return self.estimates()

    def update(self, detections, timestamp_ns):
        """Kalman-update each detection's assigned track (one detection per track); confirmed births start new tracks; anything
        else (its track vanished or was taken, or an unconfirmed birth) gets track_id None."""
        timestamp_ns = int(timestamp_ns)
        self.predict(timestamp_ns)
        assignments, taken = {}, set()
        for index, detection in enumerate(detections):
            if detection.global_track_bootstrap_only:
                detection.track_id = None
            elif detection.track_id in self.tracks and detection.track_id not in taken:
                assignments[index] = detection.track_id
                taken.add(detection.track_id)
            else:
                detection.track_id, detection.track_prediction_guided, detection.global_track_bootstrap_only = None, False, True

        for index, track_id in assignments.items():
            detection, track = detections[index], self.tracks[track_id]
            R = self.R
            innovation = detection.ground_xy - self.H @ track["state"]
            S = self.H @ track["covariance"] @ self.H.T + R
            K = track["covariance"] @ self.H.T @ np.linalg.inv(S)
            track["state"] = track["state"] + K @ innovation
            correction = np.eye(4) - K @ self.H
            track["covariance"] = correction @ track["covariance"] @ correction.T + K @ R @ K.T
            if track["missed"] <= 1 and track["consecutive_started_ns"] is not None:
                track["consecutive_hits"] += 1
            else:
                track["consecutive_hits"] = 1
                track["consecutive_started_ns"] = timestamp_ns
            track["missed"] = 0
            track["hit_count"] += 1
            track["observed"] = True
            track["last_observed_ns"] = timestamp_ns
            track["last_observed_xy"] = np.asarray(detection.ground_xy, dtype=np.float64).copy()
            self.update_metadata(track, detection)
            detection.smoothed_xy = track["state"][:2].copy()
            for member in detection.members:
                member.track_id = track_id

        for index, detection in enumerate(detections):
            if index in assignments:
                continue
            if not detection.global_birth_confirmed:
                detection.smoothed_xy = None
                for member in detection.members:
                    member.track_id = None
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            R = self.R
            track = {
                "state": np.array([detection.ground_xy[0], detection.ground_xy[1], 0.0, 0.0], dtype=np.float64),
                "covariance": np.diag([R[0, 0], R[1, 1], self.max_speed ** 2, self.max_speed ** 2]),
                "last_ns": timestamp_ns, "last_observed_ns": timestamp_ns,
                "missed": 0, "age": 1, "hit_count": 1, "consecutive_hits": 1, "consecutive_started_ns": timestamp_ns,
                "observed": True, "last_observed_xy": np.asarray(detection.ground_xy, dtype=np.float64).copy(),
                "yaw": None, "yaw_concentration": None, "spread": 0.0, "camera_indices": (),
            }
            self.update_metadata(track, detection)
            self.tracks[track_id] = track
            detection.track_id = track_id
            detection.smoothed_xy = detection.ground_xy.copy()
            for member in detection.members:
                member.track_id = track_id

        for track in self.tracks.values():
            if not track["observed"]:
                track["consecutive_hits"] = 0
                track["consecutive_started_ns"] = None
        return detections

    def estimates(self):
        return [TrackedPerson(
            track_id=track_id, position_ground_xy=track["state"][:2].copy(), velocity_ground_xy_mps=track["state"][2:4].copy(),
            ground_yaw=track["yaw"], spread_m=float(track["spread"]), yaw_concentration=track["yaw_concentration"],
            observed_this_frame=track["observed"], missed_steps=track["missed"],
            missed_seconds=max(0.0, (track["last_ns"] - track["last_observed_ns"]) / 1e9), age_steps=track["age"],
            hit_count=track["hit_count"], confirmed=True, last_observed_timestamp_ns=track["last_observed_ns"],
            camera_indices=track["camera_indices"], position_covariance=track["covariance"][:2, :2].copy(),
            consecutive_hit_count=track["consecutive_hits"], consecutive_observed_seconds=self.consecutive_seconds(track),
            last_observed_position_ground_xy=None if track["last_observed_xy"] is None else track["last_observed_xy"].copy())
            for track_id, track in sorted(self.tracks.items())]


def prediction_costs(observations, predictions, prediction_gate, noise):
    """(observations x predictions) matrix of metre-equivalent Mahalanobis distance + covariance-volume cost of each camera observation
    vs each global track prediction; NaN where the raw distance is beyond prediction_gate."""
    costs = np.full((len(observations), len(predictions)), np.nan)
    if not observations or not predictions:
        return costs
    R, measurement_logdet, scale = noise
    innovation = (np.array([o.ground_xy for o in observations], dtype=np.float64)[:, None, :]
                  - np.array([p.position_ground_xy for p in predictions], dtype=np.float64)[None, :, :])
    S = np.array([p.position_covariance for p in predictions], dtype=np.float64) + R
    determinant = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    x, y = innovation[..., 0], innovation[..., 1]
    nis = (S[:, 1, 1] * x * x - (S[:, 0, 1] + S[:, 1, 0]) * x * y + S[:, 0, 0] * y * y) / determinant  # innovation^T S^-1 innovation
    cost = scale * np.sqrt(np.maximum(0.0, nis)) + 0.5 * scale * np.maximum(0.0, np.log(determinant) - measurement_logdet)
    inside = np.sqrt(x * x + y * y) <= prediction_gate
    costs[inside] = cost[inside]
    return costs


def fuse_cluster(members, cluster_index, predicted_track_id):
    """Mean of the camera feet, with the fused yaw."""
    positions = np.stack([np.asarray(m.ground_xy, dtype=np.float64) for m in members], axis=0)
    mean = np.mean(positions, axis=0)
    residuals = np.linalg.norm(positions - mean, axis=1)
    yaw, concentration = fuse_orientations(members)
    for member in members:
        member.cluster_index = cluster_index
    return FusedPerson(
        members=members, ground_xy=mean, ground_yaw=yaw, spread_m=float(np.max(residuals)), track_id=predicted_track_id,
        yaw_concentration=concentration, raw_ground_xy=mean,
        track_prediction_guided=predicted_track_id is not None, global_track_bootstrap_only=predicted_track_id is None)


def associate_and_fuse(observations, predictions, noise, association_gate, prediction_gate, yaw_weight, min_regret):
    """Cluster current camera observations, split clusters whose members clearly prefer different global tracks,
    then assign whole clusters one-to-one to global predictions."""
    raw_clusters = complete_link_clusters(observations, association_gate, yaw_weight)
    predictions = sorted(predictions, key=lambda p: p.track_id)

    def strong_best(costs):
        costs = sorted(costs, key=lambda item: (item[0], item[1]))
        if not costs:
            return None
        best_cost, best = costs[0]
        regret = max(0.0, min([prediction_gate] + [c for c, _ in costs[1:]]) - best_cost)
        return best if best_cost < prediction_gate and regret > 1e-12 and regret + 1e-12 >= min_regret else None

    cost_matrix = prediction_costs(observations, predictions, prediction_gate, noise)
    strong_by_observation = {}
    for index in range(len(observations)):
        columns = np.flatnonzero(~np.isnan(cost_matrix[index]))
        best = strong_best([(float(cost_matrix[index, c]), predictions[c].track_id) for c in columns])
        if best is not None:
            strong_by_observation[index] = best

    clusters = []
    for cluster in raw_clusters:
        if len({strong_by_observation[i] for i in cluster if i in strong_by_observation}) <= 1:
            clusters.append(cluster)
            continue
        by_track, unclaimed = {}, []
        for i in cluster:
            if i in strong_by_observation:
                by_track.setdefault(strong_by_observation[i], []).append(i)
            else:
                unclaimed.append(i)
        clusters.extend(list(by_track.values()) + [[i] for i in unclaimed])

    assignments = {}
    if clusters and predictions:
        pair_costs = {}
        max_pair = prediction_gate + yaw_weight * (np.pi / 2.0)
        for row, cluster in enumerate(clusters):
            member_costs = cost_matrix[cluster]  # (members, predictions); a prediction needs every member inside its gate
            for column in np.flatnonzero(~np.isnan(member_costs).any(axis=0)):
                pair_costs[(row, int(column))] = float(np.mean(member_costs[:, column]))
                max_pair = max(max_pair, pair_costs[(row, int(column))])

        n_rows, n_columns = len(clusters), len(predictions)
        invalid = (max_pair + prediction_gate + 1.0) * (n_rows + n_columns + 1)
        costs = np.full((n_rows, n_columns + n_rows), prediction_gate)
        costs[:, :n_columns] = invalid
        valid = np.zeros((n_rows, n_columns), dtype=bool)
        for (row, column), cost in pair_costs.items():
            valid[row, column] = True
            costs[row, column] = cost
        rows, columns = linear_sum_assignment(costs)
        assignments = {row: predictions[column].track_id for row, column in zip(rows.tolist(), columns.tolist())
                       if column < n_columns and valid[row, column]}

    fused = [fuse_cluster([observations[i] for i in cluster], index, assignments.get(index)) for index, cluster in enumerate(clusters)]
    fused.sort(key=lambda p: (float(p.ground_xy[0]), float(p.ground_xy[1]), -1 if p.track_id is None else p.track_id))
    for index, person in enumerate(fused):
        for member in person.members:
            member.cluster_index = index
    return fused


class MultiCameraTracker:
    """Multiview clustering/assignment of camera detections to global Kalman tracks -> private 1 s birth verification of unclaimed
    clusters -> public global IDs."""

    def __init__(self, dedup_radius_m, association_gate_m, prediction_gate_m, yaw_weight, split_min_regret, birth_gate_m, birth_max_gap_s,
                 birth_min_s, coast_s, acceleration_noise, measurement_noise_m, max_speed_mps, yaw_gain):
        """Parameters: the `tracking` section of the pipeline config."""
        self.dedup_radius_m, self.coast_s = dedup_radius_m, coast_s
        self.association = dict(association_gate=association_gate_m, prediction_gate=prediction_gate_m, yaw_weight=yaw_weight,
                                min_regret=split_min_regret)
        self.births = dict(gate_m=birth_gate_m, max_gap_seconds=birth_max_gap_s, min_seconds=birth_min_s)
        self.noise = measurement_noise(measurement_noise_m)
        self.global_tracker = Tracker(self.noise[0], coast_s, acceleration_noise, max_speed_mps, yaw_gain)
        self.pending_births, self.next_birth_id = {}, 0  # private birth candidates: {id: {"xy", "started_ns", "last_ns"}}
        self.last_pending_birth_observations = []
        self.last_global_birth_promotions = []

    def verify_births(self, candidates, timestamp_ns, gate_m, max_gap_seconds, min_seconds):
        """Pending births are plain points: each candidate matches the nearest pending point within gate_m (one-to-one); a pending
        point seen in consecutive frames (gaps <= max_gap_seconds) for >= min_seconds is promoted. Unmatched candidates start new ones."""
        pending = self.pending_births
        for private_id in [p for p, b in pending.items() if (timestamp_ns - b["last_ns"]) / 1e9 > self.coast_s]:
            del pending[private_id]
        ids = sorted(pending)
        matched = {}
        if candidates and ids:
            costs = np.full((len(candidates), len(ids)), 1e6)
            for row, candidate in enumerate(candidates):
                for column, private_id in enumerate(ids):
                    distance = float(np.linalg.norm(candidate.ground_xy - pending[private_id]["xy"]))
                    if distance <= gate_m:
                        costs[row, column] = distance
            for row, column in zip(*linear_sum_assignment(costs)):
                if costs[row, column] <= gate_m:
                    matched[int(row)] = ids[int(column)]
        promoted = []
        for row, candidate in enumerate(candidates):
            if row in matched:
                birth = pending[matched[row]]
                if (timestamp_ns - birth["last_ns"]) / 1e9 > max_gap_seconds + 1e-12:
                    birth["started_ns"] = timestamp_ns
                birth["xy"], birth["last_ns"], private_id = np.asarray(candidate.ground_xy, dtype=np.float64), timestamp_ns, matched[row]
            else:
                private_id, self.next_birth_id = self.next_birth_id, self.next_birth_id + 1
                pending[private_id] = {"xy": np.asarray(candidate.ground_xy, dtype=np.float64), "started_ns": timestamp_ns, "last_ns": timestamp_ns}
            birth = pending[private_id]
            confirmed = birth["last_ns"] > birth["started_ns"] and (birth["last_ns"] - birth["started_ns"]) / 1e9 + 1e-12 >= min_seconds
            candidate.global_birth_private_track_id = private_id
            candidate.global_birth_consecutive_started_timestamp_ns = int(birth["started_ns"])
            candidate.global_birth_pending, candidate.global_birth_confirmed = not confirmed, confirmed
            for member in candidate.members:
                member.global_birth_pending, member.global_birth_private_track_id = not confirmed, private_id
            if confirmed:
                promoted.append(candidate)
                del pending[private_id]
        return promoted

    def update(self, detections, timestamp_ns):
        """One frame: all cameras' projected detections -> tracked people (FusedPerson with track_id)."""
        self.last_global_birth_promotions = []
        observations = suppress_same_camera_duplicates(detections, self.dedup_radius_m)
        for observation in observations:
            observation.camera_track_confirmed = True  # output field; detections dropped as duplicates before this stay None
        predictions = self.global_tracker.predict(timestamp_ns)
        fused = associate_and_fuse(observations, predictions, self.noise, **self.association)
        global_updates = [p for p in fused if not p.global_track_bootstrap_only]
        births = [p for p in fused if p.global_track_bootstrap_only]
        promoted = self.verify_births(births, timestamp_ns, **self.births)
        promoted_ids = {id(p) for p in promoted}
        self.last_pending_birth_observations = [p for p in births if id(p) not in promoted_ids]
        next_public_id = self.global_tracker.next_track_id
        updated = self.global_tracker.update(global_updates + promoted, timestamp_ns)
        updated = [p for p in updated if p.track_id is not None]
        for person in promoted:
            if person.track_id is None or person.track_id < next_public_id:
                continue
            self.last_global_birth_promotions.append(GlobalBirthPromotion(
                int(person.global_birth_private_track_id), int(person.track_id),
                int(person.global_birth_consecutive_started_timestamp_ns), int(timestamp_ns)))
        return updated

    def estimates(self):
        return self.global_tracker.estimates()
