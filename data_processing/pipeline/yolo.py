"""Person boxes from YOLO (a TensorRT engine or .pt weights) on all cameras of a frame at once, pre- and post-processing on the GPU."""
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
from ultralytics.nn.autobackend import AutoBackend


def load_detector(weights):
    return AutoBackend(str(weights), device=torch.device("cuda:0"), fp16=True)


def find_people(detector, pixels, image_size, min_confidence, iou=0.7):
    """pixels: (cameras, H, W, 3) uint8 BGR array in pinned memory -> per camera (boxes xyxy pixels, scores) of class 0 (person).
    Same letterbox, score threshold and NMS as ultralytics' predict."""
    images = torch.from_numpy(pixels).cuda(non_blocking=True).flip(-1).permute(0, 3, 1, 2).float().div_(255)
    height, width = images.shape[-2:]
    ratio = image_size / max(height, width)
    new_h, new_w = int(round(height * ratio)), int(round(width * ratio))
    top, left = (image_size - new_h) // 2, (image_size - new_w) // 2
    x = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
    x = F.pad(x, (left, image_size - new_w - left, top, image_size - new_h - top), value=114 / 255).half()
    output = detector(x)
    output = output[0] if isinstance(output, (list, tuple)) else output  # (cameras, 4 + classes, anchors): cx, cy, w, h, class scores
    camera, anchor = torch.nonzero(output[:, 4] > min_confidence, as_tuple=True)
    cx, cy, w, h, scores = output[camera, :5, anchor].float().unbind(dim=1)
    boxes = torch.stack((cx - w / 2 - left, cy - h / 2 - top, cx + w / 2 - left, cy + h / 2 - top), dim=1) / ratio
    keep = batched_nms(boxes, scores, camera, iou)
    camera, boxes, scores = camera[keep].cpu().numpy(), boxes[keep].double().cpu().numpy(), scores[keep].double().cpu().numpy()
    return [(boxes[camera == c], scores[camera == c]) for c in range(len(output))]
