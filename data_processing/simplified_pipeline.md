# Original vs Simplified Pipeline

Dense GDC atrium clip (701 frames, 6 cameras), scored against the reviewed ground truth. Simplified values are mean ± sd over 3 runs.

| | Original | Simplified Pipeline |
|---|---|---|
| HOTA | 96.18 | 96.71 ± 0.16 |
| IDF1 | 97.52 | 97.66 ± 0.19 |
| MOTA | 95.70 | 95.89 ± 0.06 |
| F1 | 98.22 | 98.31 ± 0.01 |
| Precision | 99.69 | 99.32 ± 0.01 |
| Recall | 96.79 | 97.31 ± 0.00 |
| ID switches | 9 | 5.7 ± 1.2 |
| Fragmentations | 80 | 47.0 ± 1.7 |
| Position error (cm) | 1.40 | 1.65 ± 0.01 |
| Yaw error (deg) | 8.60 | 6.69 ± 0.01 |
| SAM, s/frame (GPU) | 5.39 | 4.06 |
| Post-SAM, s/frame (CPU) | 0.87 | 0.22 |
| **Total, s/frame** | **6.26** | **4.28 (1.46x faster)** |
| Projected time, 10-minute rosbag (12,000 frames) | 20.9 h | 14.3 h |
| Lines of code, SAM | 369 | 395 |
| Lines of code, post-SAM | 1,565 | 937 |
| **Lines of code, total** | **1,934** | **1,332** |

## What is different

### SAM (`pipeline/infer.py`)
- SAM3 runs on all images first; then SAM 3D Body runs on all person crops, batched 256 at a time across images and frames.
- The SAM 3D Body backbone is `torch.compile`d.
- SAM3 text embeddings are computed once, and one image encoding is shared by all prompts.
- New output: scene prompts (furniture, obstacle, stairs, door, column, wall) on every 20th frame.
- Torch 2.7.1 instead of 2.5.1.
- Other inference variants (per-image, multi-GPU, SAM 3.1 video) were removed.

### Post-SAM (`pipeline/run.py`, `ground.py`, `track.py`)
Removed:
- Per-camera Kalman tracking (2-hit admission, speed gate, tentative lane).
- The private birth Kalman tracker.
- Regret re-solves, binding bonuses and duplicate vetoes in association.
- Robust (Huber) position and yaw fusion.
- Velocity-based yaw, the velocity cap, and the yaw term in track association.
- Mahalanobis gates (distance gates remain).
- The foot-quality noise model and the ankle fallback for the foot point.
- 5 of the 7 anatomy-gate tests.
- The truncated-feet birth rule.
- Repeated or dead checks, and fields that were written but never read.

Simplified:
- Birth verification: a pending birth is a plain point, promoted after 1 s of consecutive sightings.
- Duplicate suppression: greedy by confidence.
- Multi-camera fusion: plain mean of the camera feet and of their yaws.
- Measurement noise: one constant (0.35 m).
- Hungarian matching: SciPy instead of hand-written.

Unchanged: output format, rendering, SAM cache format.
