"""Pass 1 (GPU): SAM3 "person" on every image (text embeddings computed once; the image encoding is shared with the scene prompts on
every scene_every-th frame), then SAM 3D Body on all person crops, pooled across images and frames into calls of exactly body_batch
crops, with a torch.compiled backbone. Writes cache_dir/camN/TIMESTAMP.json (people), cache_dir/scene/camN/PROMPT.json
({stamp: boxes_xyxy, scores, masks_rle (COCO RLE)}) and cache_dir/timing.json. Other inference variants live in speedup_variants/."""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import frames
import sam


def encode_texts(model, prompts):
    """SAM3 text-encoder outputs per prompt, computed once (set_text_prompt would redo this for every image)."""
    torch = model["torch"]
    with torch.inference_mode(), torch.cuda.device(model["sam3_device"]), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return {prompt: model["sam3"].backbone.forward_text([prompt], device=model["sam3_device"]) for prompt in prompts}


def find_people_and_scene(model, image_bgr, texts, with_scene, scene_prompts, person_threshold, scene_threshold, box_expansion):
    """One image encoding -> "person" grounding and, if with_scene, each scene prompt grounded on its own.
    Returns (people, {prompt: (boxes xyxy pixels, scores, binary masks)})."""
    from PIL import Image

    torch = model["torch"]
    height, width = image_bgr.shape[:2]
    image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    host = lambda value: value.detach().cpu().float().numpy()
    found = {}
    with torch.inference_mode():
        processor = model["processor"](model["sam3"], device=model["sam3_device"], confidence_threshold=person_threshold)
        with torch.cuda.device(model["sam3_device"]), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(Image.fromarray(image_rgb.astype(np.uint8), "RGB"))
            state["geometric_prompt"] = model["sam3"]._get_dummy_prompt()
            state["backbone_out"].update(texts["person"])
            output = processor._forward_grounding(state)
            raw_boxes, scores = host(output["boxes"]).astype(np.float32).reshape(-1, 4), host(output["scores"]).astype(np.float32).reshape(-1)
            if with_scene:
                processor.confidence_threshold = scene_threshold
                for prompt in scene_prompts:
                    state["backbone_out"].update(texts[prompt])
                    output = processor._forward_grounding(state)
                    found[prompt] = (host(output["boxes"]).astype(np.float32).reshape(-1, 4), host(output["scores"]).astype(np.float32).reshape(-1),
                                     output["masks"].detach().cpu().numpy().reshape(-1, *image_bgr.shape[:2]))
    keep = scores > person_threshold
    return sam.people_from_boxes(raw_boxes[keep], scores[keep], width, height, box_expansion), found


def scene_entry(boxes, scores, masks):
    """One image's result for one scene prompt, as stored in cache_dir/scene/camN/PROMPT.json."""
    import pycocotools.mask as mask_utils

    return {"boxes_xyxy": boxes.round(2).tolist(), "scores": scores.round(4).tolist(),
            "masks_rle": [{"size": rle["size"], "counts": rle["counts"].decode("ascii")}
                          for rle in mask_utils.encode(np.asfortranarray(masks.transpose(1, 2, 0).astype(np.uint8)))]}


def cache_complete(groups, cameras, cache_dir, scene_prompts):
    """True if every image of groups has a cache entry and every camera has its scene files (pass 1 has nothing left to do)."""
    cache_dir = Path(cache_dir)
    return (all(sam.cache_path(cache_dir, c, stamp).is_file() for _, group in groups for c, stamp in group.items())
            and all((cache_dir / "scene" / f"cam{c}" / f"{prompt}.json").is_file() for c in cameras for prompt in scene_prompts))


def run(frames_dir, groups, cameras, cache_dir, checkpoint, mhr, sam3_checkpoint, dinov3_code, person_threshold, box_expansion, scene_prompts, scene_threshold,
        scene_every, body_batch):
    """Parameters after dinov3_code: the config's `detection` section."""
    cache_dir = Path(cache_dir)
    scene_prompts = tuple(scene_prompts)
    model = sam.load_sam_model(checkpoint, mhr, sam3_checkpoint, dinov3_code)
    sam.compile_body(model)
    seconds, clock, images_done, scene_images = defaultdict(float), time.perf_counter(), 0, 0

    def lap(stage):
        nonlocal clock
        model["torch"].cuda.synchronize()
        now = time.perf_counter()
        seconds[stage] += now - clock
        clock = now

    scene_done = {c for c in cameras if all((cache_dir / "scene" / f"cam{c}" / f"{prompt}.json").is_file() for prompt in scene_prompts)}
    scene = defaultdict(dict)  # (camera, prompt) -> {stamp: entry}
    texts = encode_texts(model, ("person",) + scene_prompts)
    lap("load_texts")
    people_by_image = {}
    for frame_number, (_, images) in enumerate(frames.iter_group_images(frames_dir, groups)):
        lap("video_decode")
        for camera_index, (stamp, image) in sorted(images.items()):
            lap("cache_check")
            people_cached = sam.cache_path(cache_dir, camera_index, stamp).is_file()
            with_scene = camera_index not in scene_done and frame_number % scene_every == 0
            if people_cached and not with_scene:
                continue
            people, found = find_people_and_scene(model, image, texts, with_scene, scene_prompts, person_threshold, scene_threshold, box_expansion)
            lap("sam3" if not with_scene else "sam3_and_scene")
            if not people_cached:
                people_by_image[(camera_index, stamp)] = people
            if with_scene:
                for prompt, (boxes, scores, masks) in found.items():
                    scene[(camera_index, prompt)][str(stamp)] = scene_entry(boxes, scores, masks)
                scene_images += 1
                lap("scene_write")

    def write(key, image, detections):
        nonlocal images_done
        camera_index, stamp = key
        sam.write_cache(sam.cache_path(cache_dir, camera_index, stamp), camera_index, stamp, sam.fingerprint(image, cameras[camera_index]["K"]), detections)
        images_done += 1

    batcher = sam.BodyBatcher(model, body_batch, write, lap)
    for _, images in frames.iter_group_images(frames_dir, groups):
        lap("video_decode")
        for camera_index, (stamp, image) in sorted(images.items()):
            lap("cache_check")
            if (camera_index, stamp) in people_by_image:
                batcher.add((camera_index, stamp), image, cameras[camera_index]["K"], people_by_image.pop((camera_index, stamp)))
    batcher.flush()

    for (camera_index, prompt), entries in sorted(scene.items()):
        path = cache_dir / "scene" / f"cam{camera_index}" / f"{prompt}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries))
    lap("scene_write")

    per_frame = {stage: total / len(groups) for stage, total in seconds.items()}
    if images_done or scene_images:  # a rerun on a complete cache keeps the timing of the run that actually inferred
        (cache_dir / "timing.json").write_text(json.dumps({
            "variant": "combined", "scene_every": scene_every, "body_batch": body_batch,
            "peak_gpu_gb": round(model["torch"].cuda.max_memory_reserved() / 2**30, 2), "frames": len(groups),
            "images_inferred": images_done, "scene_images": scene_images, "seconds_per_frame": per_frame, "total_seconds": dict(seconds)}, indent=1))
    print(f"combined: {images_done} images, {scene_images} scene images; seconds per frame: "
          + ", ".join(f"{k} {v:.3f}" for k, v in per_frame.items()), flush=True)

