#!/usr/bin/env bash
# Submit pipeline runs to TACC (Slurm), one input per GPU, GPUS_PER_NODE inputs per job. Run from a login node, in this directory:
#   scripts/tacc_submit.sh /path/to/bag_a /path/to/bag_b [...]
# Each input is a ROS 2 bag directory (or .mcap file), or an mp4 directory with frames.json. Outputs: outputs/<input name>/.
# Options (env): GPUS_PER_NODE (2), PARTITION (gpu-a100), ACCOUNT (IRI25030), TIME_LIMIT (24:00:00),
#   CONFIG (config/pipeline.yaml), SIF_PATH ($WORK/containers/gdc-pipeline.sif).
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
PARTITION="${PARTITION:-gpu-a100}"
ACCOUNT="${ACCOUNT:-IRI25030}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CONFIG="${CONFIG:-config/pipeline.yaml}"
SIF_PATH="${SIF_PATH:-$WORK/containers/gdc-pipeline.sif}"
[[ $# -gt 0 ]] || { sed -n '2,6p' "$0"; exit 2; }
for input in "$@"; do [[ -e $input ]] || { echo "Not found: $input" >&2; exit 2; }; done
mkdir -p outputs/slurm

inputs=("$@")
for ((start = 0; start < ${#inputs[@]}; start += GPUS_PER_NODE)); do
  group=("${inputs[@]:start:GPUS_PER_NODE}")
  job="gdc_$(basename "${group[0]%.mcap}")"
  if [[ ${#group[@]} -gt 1 ]]; then job+="+$((${#group[@]} - 1))"; fi
  runs=""
  for gpu in "${!group[@]}"; do
    input=$(realpath "${group[$gpu]}") name=$(basename "${input%.mcap}")
    source=$([[ -f $input/frames.json ]] && echo "--mp4-dir $input" || echo "--rosbag-dir $input")
    runs+="mkdir -p outputs/$name && CUDA_VISIBLE_DEVICES=$gpu apptainer exec --nv --bind \$WORK,\$SCRATCH,$(dirname "$input") $SIF_PATH \\
  python pipeline/pipeline.py --config $CONFIG $source --output-dir outputs/$name > outputs/$name/pipeline.log 2>&1 &
"
  done
  sbatch <<EOF
#!/bin/bash
#SBATCH -J $job
#SBATCH -o outputs/slurm/${job}_%j.out
#SBATCH -p $PARTITION
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t $TIME_LIMIT
#SBATCH -A $ACCOUNT
set -uo pipefail
module load tacc-apptainer
cd $PWD
nvidia-smi -L
$runs
failed=0
for pid in \$(jobs -p); do wait \$pid || failed=1; done
exit \$failed
EOF
done
