# GDC atrium pipeline

Six cameras -> people detection and 3D body pose (SAM3 + SAM 3D Body) -> floor positions -> multi-camera tracking -> `frames.jsonl`.
All settings (model paths, thresholds, tracking parameters) are in `config/pipeline.yaml`.

## Run
On a machine with an A100/H100 GPU, from this directory:
```bash
python pipeline/pipeline.py --config config/pipeline.yaml --rosbag-dir /path/to/bag --output-dir outputs/myrun
python pipeline/pipeline.py --config config/pipeline.yaml --mp4-dir /path/to/videos --output-dir outputs/myrun
```
- `--rosbag-dir`: a ROS 2 bag directory (or one `.mcap` file). Its camera videos are extracted to `outputs/myrun/videos` first;
  add `--extract-only` to stop there.
- `--mp4-dir`: previously extracted videos (`cam0..5.mp4` + `frames.json`).
- `--max-frames N`: only the first N frames (quick test).

Rerunning the same command reuses the GPU results in `outputs/myrun/cache`.

## Run on TACC
One input per GPU, two GPUs per node (edit `GPUS_PER_NODE` for other nodes):
```bash
scripts/tacc_submit.sh /path/to/bag_a /path/to/bag_b
```
Outputs go to `outputs/bag_a/` and `outputs/bag_b/` (with `pipeline.log`); Slurm logs to `outputs/slurm/`. Partition, account,
time limit and container path are set at the top of the script. This directory must be on `$WORK` or `$SCRATCH`.

## Output
`outputs/myrun/frames.jsonl`: one line per synchronized frame (20 Hz) with `frame_index`, `timestamp_ns`, `present_cameras`,
`source_timestamp_ns_by_camera`, `sync_skew_ns`, `model_detections_by_camera` (boxes, confidences, 2D/3D keypoints),
`projected_detections` (floor positions), `fused_people`, `tracked_people` (track IDs, positions, headings, velocities) and
`scene_detections_by_camera` (furniture, obstacles, stairs, doors, columns and walls; latest result per camera).
Also camera and bird's-eye videos (`output.render` in the config) and `timing.json`.

## Setup
Container (recommended; required on TACC):
```bash
cd container && docker build -f Containerfile -t <dockerhub-user>/gdc-pipeline:latest . && docker push <dockerhub-user>/gdc-pipeline:latest
# on TACC:
module load tacc-apptainer && apptainer pull $WORK/containers/gdc-pipeline.sif docker://<dockerhub-user>/gdc-pipeline:latest
```
Without the container (Python 3.11, CUDA GPU; system packages ffmpeg, libgl1, libegl1):
```bash
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r container/requirements.txt
git clone https://github.com/facebookresearch/sam3.git && git -C sam3 checkout 5dd401d1c5c1d5c3eedff06d41b77af824517619
pip install 'setuptools<70' && pip install -e sam3
```
Model weights are not in git; put them in `models/` next to this README (paths in the config):
```
models/sam3/sam3.pt                                   SAM3 (huggingface.co/facebook/sam3)
models/sam-3d-body-dinov3/model.ckpt, model_config.yaml   SAM 3D Body (huggingface.co/facebook/sam-3d-body-dinov3)
models/sam-3d-body-dinov3/assets/mhr_model.pt
models/dinov3_code/                                   DINOv3 code (github.com/facebookresearch/dinov3)
```
The SAM3 and SAM 3D Body weights are under Meta's licenses (gated on Hugging Face).

## Check against the reviewed ground truth (dense test clip; needs `data/dense_35s.mcap` and `data/gt/`)
```bash
python pipeline/pipeline.py --config config/pipeline.yaml --rosbag-dir data/dense_35s.mcap --output-dir outputs/dense
scripts/evaluate.sh dense
```
Expect HOTA about 96.7 (`simplified_pipeline.md` compares this pipeline with the original one).
