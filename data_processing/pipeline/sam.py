import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ground import box_quality_ok

SPECIAL = "__atrium_cache_value__"
MHR70_FOOT_CONTACT = {
    "left_indices": [15, 16, 17], "right_indices": [18, 19, 20], "fallback_indices": [13, 14], "pooled_indices": [],
    "contact_source": "toe_heel_midpoint", "contact_quality": 1.0, "fallback_source": "ankles", "fallback_quality": 0.7,
}


@dataclass
class RawDetection:
    bbox_xyxy: np.ndarray
    keypoints_2d: np.ndarray
    keypoints_3d: np.ndarray = None
    global_rot_zyx: np.ndarray = None
    confidence: float = 1.0
    instance_mask: np.ndarray = None
    mask_confidence: float = None
    model_fields: dict = field(default_factory=dict)


def from_json(value):
    if isinstance(value, list):
        return [from_json(item) for item in value]
    if not isinstance(value, dict):
        return value
    special = value.get(SPECIAL)
    if special == "nan":
        return float("nan")
    if special in ("positive_infinity", "negative_infinity"):
        return float("inf") if special == "positive_infinity" else float("-inf")
    if special == "ndarray":
        return np.asarray(from_json(value["data"]), dtype=np.dtype(value["dtype"]))
    return {key: from_json(item) for key, item in value.items()}


def to_json(value):
    if isinstance(value, np.ndarray):
        return {SPECIAL: "ndarray", "dtype": str(value.dtype), "shape": list(value.shape), "data": to_json(value.tolist())}
    if isinstance(value, np.generic):
        return to_json(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return {SPECIAL: "nan" if math.isnan(value) else ("positive_infinity" if value > 0 else "negative_infinity")}
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    return value


def optional_array(value, shape=None):
    if value is None:
        return None
    array = np.asarray(from_json(value), dtype=np.float32)
    return array if shape is None else array.reshape(shape)


def fingerprint(image_bgr, K):
    digest = hashlib.sha256()
    digest.update(str(image_bgr.shape).encode("ascii"))
    digest.update(str(image_bgr.dtype).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(image_bgr)).cast("B"))
    digest.update(memoryview(np.asarray(K, dtype="<f8", order="C")).cast("B"))
    return digest.hexdigest()


def cache_path(cache_dir, camera_index, timestamp_ns):
    return Path(cache_dir) / f"cam{camera_index}" / f"{timestamp_ns}.json"


def read_cache(path, image_fingerprint):
    document = json.loads(path.read_text())
    if document["input_fingerprint"] != image_fingerprint:
        raise RuntimeError(f"cache entry {path} was made from a different image/K")
    return [
        RawDetection(
            bbox_xyxy=optional_array(item["bbox_xyxy"], 4),
            keypoints_2d=optional_array(item["keypoints_2d"]),
            keypoints_3d=optional_array(item.get("keypoints_3d")),
            global_rot_zyx=optional_array(item.get("global_rot_zyx"), 3),
            confidence=float(item.get("confidence", 1.0)),
            model_fields=from_json(item.get("model_fields", {})),
        )
        for item in document["detections"]
    ]


def write_cache(path, camera_index, timestamp_ns, image_fingerprint, detections):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 2, "camera_index": camera_index, "timestamp_ns": timestamp_ns,
        "input_fingerprint": image_fingerprint, "backend_signature": "simple",
        "detections": [{
            "bbox_xyxy": to_json(d.bbox_xyxy), "keypoints_2d": to_json(d.keypoints_2d),
            "keypoints_3d": to_json(d.keypoints_3d), "global_rot_zyx": to_json(d.global_rot_zyx),
            "confidence": d.confidence, "instance_mask": None, "mask_confidence": None,
            "model_fields": to_json(d.model_fields),
        } for d in detections],
    }, separators=(",", ":")))


def load_body_model(checkpoint_path, mhr_path, dinov3_code):
    """Load SAM 3D Body on the current cuda device; its DINOv3 backbone code comes from the local dinov3_code folder."""
    import torch
    import sam_3d_body

    torch_hub_load = torch.hub.load

    def load_local_dinov3(repo, name, *args, **kwargs):
        if repo == "facebookresearch/dinov3":
            repo, kwargs["source"] = str(dinov3_code), "local"
        return torch_hub_load(repo, name, *args, **kwargs)

    torch.hub.load = load_local_dinov3
    body_model, body_config = sam_3d_body.load_sam_3d_body(str(checkpoint_path), device="cuda", mhr_path=str(mhr_path))
    torch.hub.load = torch_hub_load
    estimator = sam_3d_body.SAM3DBodyEstimator(sam_3d_body_model=body_model, model_cfg=body_config, human_detector=None)
    return {"torch": torch, "estimator": estimator}


def load_sam_model(checkpoint_path, mhr_path, sam3_checkpoint, dinov3_code, sam3_device="cuda"):
    """Load SAM 3D Body (current cuda device) and the SAM3 image model (sam3_device) from local files."""
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = load_body_model(checkpoint_path, mhr_path, dinov3_code)
    with torch.cuda.device(sam3_device):
        sam3 = build_sam3_image_model(checkpoint_path=str(sam3_checkpoint), load_from_HF=False)
    return {**model, "sam3": sam3, "processor": Sam3Processor, "sam3_device": sam3_device}


def people_from_boxes(raw_boxes, scores, width, height, box_expansion):
    """Detector boxes (xyxy pixels) + scores -> box_expansion x expanded boxes -> box gate; returns boxes sorted in body-model order."""
    centers = (raw_boxes[:, :2] + raw_boxes[:, 2:]) * 0.5
    sizes = (raw_boxes[:, 2:] - raw_boxes[:, :2]) * box_expansion
    boxes = np.concatenate((centers - sizes * 0.5, centers + sizes * 0.5), axis=1).reshape(-1, 4)
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0.0, float(width))
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0.0, float(height))
    indices = np.flatnonzero((boxes[:, 2] - boxes[:, 0] > 1.0) & (boxes[:, 3] - boxes[:, 1] > 1.0) & np.isfinite(boxes).all(axis=1))
    order = np.lexsort((boxes[indices, 3], boxes[indices, 2], boxes[indices, 1], boxes[indices, 0]))
    indices = indices[order]
    indices = np.asarray([i for i in indices if box_quality_ok(raw_boxes[i], width, height)], dtype=np.int64)
    order = np.lexsort((boxes[indices, 3], boxes[indices, 2], boxes[indices, 1], boxes[indices, 0]))
    return {"raw_boxes": raw_boxes, "scores": scores, "boxes": boxes, "indices": indices[order]}


def estimate_bodies(model, crops):
    """One SAM 3D Body call over crops from any images/cameras: [(image_rgb, K, box_xyxy)] -> one output dict per crop (same order).
    Each crop is its own batch entry with its own K."""
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to

    torch, estimator = model["torch"], model["estimator"]
    pieces = []
    for image_rgb, K, box in crops:
        piece = recursive_to(prepare_batch(image_rgb, estimator.transform, np.asarray(box, dtype=np.float32)[None], None, None), "cuda")
        piece["cam_int"] = torch.as_tensor(K, dtype=torch.float32)[None].to(piece["img"]).clone()
        pieces.append(piece)
    batch = {key: torch.cat([p[key] for p in pieces]) if torch.is_tensor(pieces[0][key]) else
             (sum((p[key] for p in pieces), []) if isinstance(pieces[0][key], list) else pieces[0][key]) for key in pieces[0]}
    with torch.no_grad():
        estimator.model._initialize_batch(batch)
        out = estimator.model.run_inference(crops[0][0], batch, inference_type="body", transform_hand=estimator.transform_hand,
                                            thresh_wrist_angle=estimator.thresh_wrist_angle)["mhr"]
    out = recursive_to(recursive_to(out, "cpu"), "numpy")
    boxes = batch["bbox"].cpu().numpy()
    return [{"bbox": boxes[i, 0], "focal_length": out["focal_length"][i], "pred_keypoints_3d": out["pred_keypoints_3d"][i],
             "pred_keypoints_2d": out["pred_keypoints_2d"][i], "pred_cam_t": out["pred_cam_t"][i], "global_rot": out["global_rot"][i]}
            for i in range(len(crops))]


def compile_body(model):
    """torch.compile the SAM 3D Body backbone with fixed shapes (BodyBatcher pads every call to the same batch size)."""
    body = model["estimator"].model
    body.backbone = model["torch"].compile(body.backbone, dynamic=False)


class BodyBatcher:
    """Pools person crops across images AND frames into SAM 3D Body calls of exactly `size` crops (the last call is padded to `size`
    with repeats whose outputs are dropped, so the compiled backbone always sees one shape). done(key, image_bgr, detections) is
    called per image, oldest first, once all its crops are done. lap(stage) times each call as "sam3d_body" (the first call, which
    compiles, as "load_compile") and the done callbacks as "cache_write"."""

    def __init__(self, model, size, done, lap):
        self.model, self.size, self.done, self.lap = model, size, done, lap
        self.pending, self.crops, self.compiling = [], [], True  # pending: [key, image, people, outputs]; crops: [(entry, crop)]

    def add(self, key, image_bgr, K, people):
        entry = [key, image_bgr, people, []]
        self.pending.append(entry)
        image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        self.crops.extend((entry, (image_rgb, K, people["boxes"][i])) for i in people["indices"])
        while len(self.crops) >= self.size:
            self._run(self.size)

    def flush(self):
        self._run(len(self.crops))  # the remainder (also flushes images without people)

    def _run(self, count):
        part, self.crops = self.crops[:count], self.crops[count:]
        batch = [crop for _, crop in part]
        if batch:
            batch += batch[-1:] * (self.size - len(batch))
        for (entry, _), output in zip(part, estimate_bodies(self.model, batch) if batch else []):
            entry[3].append(output)
        self.lap("load_compile" if self.compiling else "sam3d_body")
        self.compiling = False
        while self.pending and len(self.pending[0][3]) == len(self.pending[0][2]["indices"]):
            key, image_bgr, people, outputs = self.pending.pop(0)
            self.done(key, image_bgr, detections_from_outputs(outputs, people))
        self.lap("cache_write")


def detections_from_outputs(outputs, people):
    """SAM 3D Body outputs (one per box of people_from_boxes, in its order) -> sorted RawDetections with detector metadata."""
    raw_boxes, scores, boxes, indices = people["raw_boxes"], people["scores"], people["boxes"], people["indices"]
    detections = []
    for position, out in enumerate(outputs):
        model_fields = {"pred_cam_t": np.asarray(out["pred_cam_t"]).copy(), "focal_length": np.asarray(out["focal_length"]).item()
                        if np.asarray(out["focal_length"]).ndim == 0 else np.asarray(out["focal_length"]).copy(),
                        "sam3d_detection_index": position, "confidence_source": "upstream_not_provided_default_1.0",
                        "keypoint_schema": "mhr70", "foot_contact": dict(MHR70_FOOT_CONTACT)}
        rot = np.asarray(out["global_rot"], dtype=np.float32).reshape(-1)
        detections.append(RawDetection(
            bbox_xyxy=np.asarray(out["bbox"], dtype=np.float32).reshape(-1).copy(),
            keypoints_2d=np.asarray(out["pred_keypoints_2d"], dtype=np.float32).copy(),
            keypoints_3d=np.asarray(out["pred_keypoints_3d"], dtype=np.float32).copy(),
            global_rot_zyx=rot.copy() if rot.shape == (3,) and np.isfinite(rot).all() else None,
            model_fields=model_fields))
    detections.sort(key=lambda d: (*d.bbox_xyxy.tolist(),))
    for detection, index in zip(detections, indices):
        detection.confidence = float(scores[index])
        detection.model_fields["confidence_source"] = "sam3_detector_score"
        detection.model_fields["proposal_source"] = "sam3_external_person_proposals"
        detection.model_fields["detector_metadata"] = {
            "detector_index": int(index), "detector_score": float(scores[index]),
            "raw_bbox": raw_boxes[index].astype(np.float32, copy=True), "expanded_bbox": boxes[index].astype(np.float32, copy=True)}
    return detections
