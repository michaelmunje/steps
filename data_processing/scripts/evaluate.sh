#!/usr/bin/env bash
# Score outputs/NAME against the reviewed dense ground truth (published settings) and print scores + seconds per frame.
#   scripts/evaluate.sh NAME [NAME ...]        (inside the container, or anywhere with python3: the evaluator is stdlib-only)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
reports=()
mkdir -p "$root/outputs/eval"
for name in "$@"; do
  for policy in all-emitted annotator-observed; do
    report="$root/outputs/eval/${name}_$policy"
    rm -rf "$report" "$report.inputs"
    python3 "$root/pipeline/evaluate.py" --gt "$root/data/gt/nathan_dense_30s.icra.json" --pred "$root/outputs/$name" --name "${name}_$policy" \
      --output "$report" --policy $policy --start-s 0 --end-s 30 --source-fps 20 --frame-step 2 --roi -4 14 -3 15 \
      --max-distance-m 1 --similarity linear --similarity-scale-m 1 > "$report.log" 2>&1
    reports+=("$report")
  done
done
python3 "$root/pipeline/metrics_table.py" "${reports[@]}"
