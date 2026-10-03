# World-position tracking evaluation

Run from `/home/zhengpengen/gdc_atrium` using Python 3.10 or newer. The evaluator works with the standard library. If SciPy is already installed, it uses its Hungarian solver; otherwise it uses the included rectangular Hungarian implementation. The selected backend is recorded.

These scripts read source predictions and reviewed GT exports without changing annotations, review flags, tracker IDs, source files, or application code. Every run creates a **new** output directory containing `report.json` and `report.md`; an existing output path is refused.

The comparison scripts and provisional light-bag performance run are complete. Dense now has a reviewed 301-frame ICRA export and a completed comparison of cleaned SAM3 against the preferred regenerated DeepStream CSV. No evaluation across all bags was run.

## Dense, first 30 seconds: primary pipeline comparison

The [current detailed report](/robodata/gdc_atrium/ICRA_GDC/dense_pipeline_eval_20260915/README.md) evaluates `nathan_dense_30s` at 10 Hz, source indices `0,2,...,600`, within inclusive ICRA X `[-4,14]`, Y `[-3,15]` metres, using a 1 m gate. All 301 requested GT frames and 12,338 GT observations are retained. It includes figures, per-frame and per-person diagnostics, frozen inputs, and reproduction manifests.

| Primary output | HOTA | IDF1 | CLEAR MOTA | CLEAR IDSW | CLEAR Frag | Base position MAE | Yaw MAE / matched-heading coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cleaned SAM3, all-emitted | 96.813% | 97.712% | 95.461% | 0 | 39 | 0.013065 m | 3.379° / 100% |
| Preferred DeepStream CSV | 26.198% | 39.770% | 24.275% | 422 | 263 | 0.394928 m | 59.435° / 28.064% |

Base distance-assignment TP/FP/FN are **12,018/116/320** for SAM3 and **8,110/4,123/4,228** for DeepStream. The separate CLEAR assignment has TP/FP/FN **11,956/178/382** and **7,825/4,408/4,513** respectively; do not combine those CLEAR MOTA/IDSW values with base counts in the MOTA formula. HOTA, IDF1, and base position/heading calculations are unchanged by adding CLEAR.

The primary SAM3 policy is **all-emitted**, retaining 175 stale published states that `annotator-observed` excludes. DeepStream also retains published held states; neither side receives an added confidence or age filter. Their native retention behavior is still different. This is a finalized-output comparison, including SAM3 retroactive states and offline heading cleanup, not a causal online benchmark. The [frozen SAM3 reproduction manifest](/robodata/gdc_atrium/ICRA_GDC/dense_pipeline_eval_20260915/reproduce_sam3_cleaned.json) reproduces the primary input.

DeepStream has 300 exact source-timestamp matches: GT frame 0 predates its first tick and is explicitly scored as unavailable empty output; GT frame 2 is a processed-empty tick. A separate 300-frame overlap sensitivity is included. DeepStream yaw is used as supplied, with unknown headings excluded. On the same **2,268** GT observations with valid matched headings from both systems, MAE is **4.063°** for SAM3 and **59.402°** for DeepStream.

GT largely inherits SAM3 positions: the small SAM3 position error measures agreement with those annotations, not independently surveyed centimetre accuracy. Preferred DeepStream calibration is producer-attested; its exact run calibration bundle is unavailable. The earlier [SAM3-only report](/robodata/gdc_atrium/ICRA_GDC/sam3_metrics_20260915/README.md) and current observed-only run remain labeled sensitivity results. In that old report, **58 IDSW** and **96.247% MOTA** were the legacy base assignment values, not CLEAR results. GT38 at source frames 218–600 remains a heading disagreement for camera adjudication.

## Example: light, first 60 seconds

```bash
python3 -B -m tracking_evaluation compare \
  --manifest tracking_evaluation/examples/light_60s_performance.json \
  --label 'provisional performance check' \
  --output /tmp/atrium-light-comparison-example
```

Choose a new output path for each run. The example reads the original SAM3D `frames.jsonl` through the annotator source directory and checks its source fingerprint against the copied GT. It selects reviewed-valid GT snapshots at source indices `0,4,...,1200` (5 Hz on the 20 Hz source clock), inclusive of 60 seconds, and uses the full scene. Missing exported GT frames are reported as gaps and excluded from both sides of evaluation. Predictions at those unlabelled frames do not become false positives.

The example remains provisional: light GT12's identity/position transitions around 3.6–6.4 seconds need adjudication. The user authorized implementation and a performance check while those issues remain. The [annotation audit](../investigations/annotation_baselines_20260914/README.md) records that historical state; its then-empty dense export has since been superseded by the reviewed 301-frame export described above.

The completed [example report](../investigations/annotation_baselines_20260914/evaluation_runs/light_60s_performance_20260914/report.md) evaluated **299 of 301** requested grid frames; 4.2 s and 57.2 s were absent from the reviewed-valid export. Runtime was **94.324 s total**, including **92.882 s loading predictions** and **0.191 s computing metrics**, with **39.52 MiB** peak RSS using the standard-library solver. It read 720,017,124 bytes of original JSONL. This one run includes filesystem/cache conditions at the time; it is not a cold-cache storage benchmark.

Provisional historical results: HOTA 0.974423, IDF1 0.989151, legacy base MOTA 0.978326, MOTP 0.003549 m, TP/FP/FN 3966/39/48, and zero base ID switches. This historical artifact predates the separate CLEAR calculation. Circular yaw MAE is 0 degrees over 3,966 known-heading matches. Those matched headings equal the stored source headings; this does not establish independently corrected heading accuracy. Nine GT observations have unknown yaw, and the broader GT audit remains unresolved.

Single-sequence invocation:

```bash
python3 -B -m tracking_evaluation compare \
  --gt /path/to/session/ground_truth.json \
  --pred /path/to/original/sam3d/output \
  --name example --start-s 0 --end-s 30 \
  --source-fps 20 --frame-step 2 \
  --max-distance-m 1 --similarity linear --similarity-scale-m 1 \
  --output /tmp/another-new-comparison
```

The source-frame step controls sampling, with frame zero as the default origin. `--frame-origin` changes the grid phase. `--source-fps` requests coverage accounting on the explicitly zero-based source clock and validates that clock against GT timestamps. It does not resample or interpolate annotations. Omit time-window options for frame-index-only normalized inputs.

Use `--roi -4 14 -3 15` for inclusive ICRA bounds. Without an ROI, all supplied world positions are considered. Cropping applies independently to GT and predictions before matching; the report includes how many observations it removed. Scope must be held constant across comparisons.

## Metrics and matching protocol

Observations use world-ground `(x,y)` in metres and optional yaw in radians, zero along +X and positive counterclockwise toward +Y. Arbitrary GT and prediction ID strings are **separate namespaces**. A prediction is never deemed correct because its ID text equals a GT ID, and prediction IDs are never repaired or remapped by the evaluator.

| Output | Definition |
| --- | --- |
| TP, FP, FN; matched observations | Per-frame one-to-one Hungarian assignment, maximum valid cardinality then minimum total Euclidean distance, with an inclusive `max_distance_m` gate |
| Precision / recall | `TP/(TP+FP)` / `TP/(TP+FN)` |
| `clear.TP`, `clear.FP`, `clear.FN` | Separate TrackEval CLEAR correspondence assignment, favoring eligible previous-timestep matches |
| `clear.IDSW` | A GT person's CLEAR prediction ID changes from its last successful match, including gaps |
| `clear.Frag` | Tracking-segment fragmentation using the pinned TrackEval CLEAR bookkeeping |
| `clear.MOTA` | `1 - (clear.FP + clear.FN + clear.IDSW) / total_GT_observations`, with reference empty-sequence conventions |
| Top-level `IDSW`, `MOTA` | Legacy base distance-assignment switch count and count-formula MOTA, retained only for JSON compatibility; these are not CLEAR results |
| MOTP_m | Mean Euclidean ground-plane distance over base spatial matches; lower is better |
| `clear.MOTP_m`, `clear.MOTP_similarity` | Mean distance and mean similarity over the separate CLEAR matches; do not substitute these for base position MAE |
| IDTP, IDFP, IDFN, IDF1 | Global one-to-one trajectory correspondence using all distance-eligible co-occurrences; `IDF1=2*IDTP/(2*IDTP+IDFP+IDFN)` |
| HOTA, DetA, AssA | Standard HOTA global-alignment procedure with world-distance similarity, averaged over 19 alpha thresholds from 0.05 to 0.95 |
| yaw_MAE_rad / yaw_MAE_deg | Mean absolute shortest circular heading difference on base matches with both headings known |
| yaw_coverage | Known-heading matched pairs divided by base TP; unknown yaw is never substituted with zero |

Human-facing CLI Markdown and stdout display **CLEAR MOTA, CLEAR IDSW, and CLEAR fragmentation**. Read `report.json` fields `combined.clear` or `sequences[i].metrics.clear` for those values. The old top-level `MOTA` and `IDSW` remain compatibility fields for the base assignment, which has no previous-frame preference. HOTA, IDF1, and base position/heading errors each retain their existing assignments and formulas.

CLEAR uses the pinned TrackEval continuity preference with similarity `max(0, 1 - distance_m / (2 * max_distance_m))`, zero outside the gate, and threshold `0.5`. This preserves the inclusive world-distance gate. Its bookkeeping, including gaps, fragmentation, and empty-side frames, has been differentially checked against the official implementation. Its association preference can produce different TP/FP/FN from maximum-cardinality base matching. The separate CLEAR similarity is not the configured HOTA similarity, and neither is image IoU.

HOTA similarity is configurable:

- `linear`: `max(0, 1 - distance_m / similarity_scale_m)`.
- `gaussian`: `exp(-0.5 * (distance_m / similarity_scale_m)^2)`.
- Both are zero outside `max_distance_m`. HOTA alpha filtering therefore changes the effective distance acceptance inside that outer gate. The base and IDF1 distance gate remains inclusive and independent of HOTA alphas.

HOTA first accumulates global alignment over a whole sequence, solves one Hungarian assignment per frame on alignment times similarity, then filters those assigned pairs at each alpha. `HOTA` is the mean of per-alpha square roots, not the square root of averaged DetA and AssA. These are **world-position HOTA scores**, not directly interchangeable with image-IoU benchmark scores. Reference procedures: [official HOTA implementation](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/hota.py), [official Identity implementation](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py), and [HOTA paper](https://arxiv.org/abs/2009.07736).

Across sequences, identities stay sequence-local. Counts and distance/angle error sums are added before ratios are computed; HOTA association accuracy is weighted by per-alpha TP before recomputing combined HOTA. CLEAR aggregates its own counts separately. The JSON contains all additive statistics, per-alpha rows, and per-sequence results. Undefined base non-HOTA ratios are `null`; HOTA and CLEAR follow their documented reference empty-denominator conventions. Scores are fractions, and MOTA can be negative.

## Input policies and provenance

GT input is the reviewed export's authoritative `ground_truth` array, including manually added missed people. Person review flags do not filter it. Extra-source annotations are not GT people, but their source predictions remain eligible to count as false positives. No source-to-GT correspondence fields are used to match identities.

SAM3D reads a contiguous JSONL prefix from frame zero through the greatest requested frame (plus a boundary record when available), and retains only requested frames in memory. Source nanosecond timestamps must exactly match GT capture timestamps. Canonical comparison time is `frame_index / 20`, matching this annotator's source convention. It checks file signatures before and after reading and reports bytes/records read and source fingerprint validation.

Two explicit prediction policies are available:

- `--sam3-policy annotator-observed` (default): the annotator's original input policy, observed tracked people plus finalized retroactive births. Repeated fused representations of an emitted ID are not duplicate people.
- `--sam3-policy all-emitted`: also include unobserved/stale tracked states. This evaluates a broader tracker-output policy; results must not be mixed with the default policy without labeling that change.

The CLI's compatibility default remains `annotator-observed`. The current dense cross-pipeline primary run explicitly uses cleaned **all-emitted** states; the observed-only input is a sensitivity. Use its frozen reproduction manifest above to reproduce the primary score, or explicitly request `--sam3-policy all-emitted` when loading the source.

Neither policy consults the annotations to remove bad predictions. Exclusion counts and reasons are recorded. Invalid emitted positions, duplicate IDs, missing prediction records, or a source fingerprint mismatch fail explicitly. An actual prediction frame with an empty list is valid and can produce false negatives; an absent prediction record is not silently treated as empty.

For the published `*.frames.cleaned.jsonl`, select `--prediction-format sam3-cleaned` and supply `--sam3-original-source` pointing to the **same original source directory used by the annotator**. This computes and verifies the original comparator fingerprint against GT; it does not give the cleaned file the original file's identity. Reports identify the cleaned product separately with `derived_fingerprint`, its publication receipt, cleanup configuration and file signatures.

The adapter requires the publisher's `.report.json` and `.receipt.json` sidecars (inferred beside the JSONL, or supplied via `--sam3-cleanup-report` / `--sam3-cleanup-receipt`). It verifies the original and published file stats, the cleanup report digest, declared frame counts, and exact requested-frame capture timestamps. Every emitted cleaned ID and XY must match an original observation in that frame. It uses cleaned `yaw_ground_rad` directly, retaining the cleanup's 180-degree corrections without applying GT edits. Only the requested prefixes are read; full multi-gigabyte data hashes are not recomputed. Reported hashes from cleanup are historical provenance, not a claim of a fresh whole-file digest check.

Cleaned predictions are already cropped, so this adapter requires an explicit evaluation ROI contained in the cleanup bounds. The dense cleanup declares ICRA **X [-4,14], Y [-3,15] m**. Use the same ROI for the original SAM3 and DeepStream comparison. An unsupported alternate JSONL can still be loaded without original provenance for a compatible caller-supplied normalized baseline; it cannot bypass the original fingerprint check against an annotator export.

The earlier [dense cleaned manifest](examples/dense_30s_cleaned_icra.json) uses the prepared 301-frame ICRA export, the published dense cleanup and explicit original-source provenance. Its default observed-only policy reproduces that sensitivity, not the current all-emitted primary input:

```bash
python3 -B -m tracking_evaluation compare \
  --manifest tracking_evaluation/examples/dense_30s_cleaned_icra.json \
  --output /tmp/dense-cleaned-comparison
```

Equivalent single-sequence invocation, with an ICRA export from `nathan_dense_30s`:

```bash
python3 -B -m tracking_evaluation compare \
  --gt /path/to/nathan_dense_30s/ground_truth.icra.10hz.0-30s.json \
  --pred /robodata/gdc_atrium/ICRA_GDC/gdc_20260824_120001/gdc_20260824_120001_bag.frames.cleaned.jsonl \
  --prediction-format sam3-cleaned \
  --sam3-original-source /robodata/gdc_atrium/ICRA_GDC/gdc_20260824_120001/gdc_20260824_120001_bag \
  --name nathan_dense_30s_sam3_cleaned --start-s 0 --end-s 30 \
  --source-fps 20 --frame-step 2 --roi -4 14 -3 15 \
  --output /tmp/dense-cleaned-comparison
```

Use a new output directory for each run. `--sam3-policy` works the same way for original and cleaned inputs. Compare them as separately named prediction variants, holding GT, ROI, sampling, policy and metric configuration constant.

## Multiple sequences and portable inputs

A manifest has a `sequences` array, optional global `metric_config`, `selection`, `roi`, and `notes`. Each sequence supplies a unique `name`, `ground_truth`, `predictions`, and `prediction_format` (`sam3`, `sam3-cleaned`, `normalized`, `deepstream`, or `deepstream-csv`). A cleaned entry also supplies `sam3_original_source` and can override `sam3_cleanup_report` / `sam3_cleanup_receipt`. A DeepStream CSV entry supplies `deepstream_frame_ledger` and `alignment`. Per-sequence selection/scope overrides global defaults; explicit CLI selection/metric options override the manifest. Relative input, ledger, and alignment paths resolve beside the manifest.

Alternatively, a manifest entry can supply `normalized_sequence` pointing to:

```json
{
  "frames": [
    {
      "frame_index": 0,
      "timestamp_s": 0.0,
      "gt": [{"track_id": "person-A", "x": 1.0, "y": 2.0, "yaw": 3.12}],
      "pred": [{"track_id": "tracker-900", "x": 1.1, "y": 2.0, "yaw": -3.12}]
    },
    {"frame_index": 1, "timestamp_s": 0.05, "gt": [], "pred": []}
  ]
}
```

Separate normalized GT/prediction JSON files use the same wrapper with only their respective `gt` or `pred` arrays. Timestamp may be `null` for index-only evaluation; yaw may be `null`. IDs must be nonempty strings in combined normalized inputs. Duplicate IDs within a side/frame are invalid. The caller supplies the common coordinate/time convention for generic normalized data.

## DeepStream CSV and frame ledger

The preferred dense producer files are `/robodata/gdc_atrium/ICRA_GDC/deepstream/dense_first90s_20hz_tracks.csv` and `dense_first90s_20hz_frames.csv`. Use the [dense CSV example](examples/dense_30s_deepstream_csv_icra.json), which reads those actual producer files and the published frozen GT/alignment:

```bash
python3 -B -m tracking_evaluation compare \
  --manifest tracking_evaluation/examples/dense_30s_deepstream_csv_icra.json \
  --output /tmp/dense-deepstream-csv-comparison
```

For a single sequence, select `--prediction-format deepstream-csv` and supply both `--deepstream-frame-ledger /path/to/frames.csv` and `--alignment /path/to/alignment.json`. The track header must be exactly `stamp_ns,track_id,x,y,yaw_rad,orient_valid,score`. The ledger supplies `frame_idx,stamp_ns` followed by each camera's `camN_header_ns,camN_record_ns,camN_repeated`, for cameras 0–5.

The [published alignment declaration](/robodata/gdc_atrium/ICRA_GDC/dense_pipeline_eval_20260915/deepstream_alignment.json) records the dataset, explicit missing-frame policy, calibration status, and these exact contracts:

```json
{
  "sequence_id": "dense-20260824-120001",
  "absence_policy": "empty",
  "coordinate_system": {"plane": "BEV", "reference": "gdc_atrium", "units": "metres"},
  "clock_alignment": {
    "tracks_stamp": "frame_ledger.stamp_ns",
    "ledger_stamp": "cam0_record_ns",
    "gt_source_stamp": "cam0_header_ns"
  },
  "calibration_provenance": {
    "status": "producer-attested",
    "description": "Producer declares the shared atrium ground frame; exact run calibration bytes are unavailable."
  }
}
```

The adapter validates the **complete files**, including records outside selected GT, hashes both files, and checks signatures before and after reading. It preserves exact integer nanoseconds and IDs. Track stamps join the ledger's cam0 receipt stamps; the default clock contract requires GT source stamps to equal the cam0 image header exactly. There is no nearest-neighbor fallback, affine clock guess, endpoint clamping, or interpolation. An existing ledger tick with no tracks is processed-empty; a missing tick is unavailable and fails by default. Explicit `absence_policy: "empty"` retains unavailable GT frames with empty predictions and reports their indices. Dataset identity must agree with the GT session.

A non-reference camera may have both header and receipt timestamps blank only when `repeated=1` follows a preceding ledger frame. The loader preserves that pair as null and reports the events and per-camera counts. It never fills them from an earlier image. Cam0 timestamps remain mandatory; unpaired blanks, unflagged blanks and malformed values are rejected. Known camera clocks must remain monotonic across unknown entries.

Some synchronized source groups use another camera's earlier image as their canonical GT timestamp. For these, the alignment declaration can supply a `source_clock_map` JSON path, resolved relative to the alignment file, and this explicit clock contract:

```json
{
  "source_clock_map": "source_clock_map.json",
  "clock_alignment": {
    "tracks_stamp": "frame_ledger.stamp_ns",
    "ledger_stamp": "cam0_record_ns",
    "gt_source_stamp": "source_clock_map.canonical_source_timestamp_ns",
    "mapped_source_stamp": "source_clock_map.cam0_header_ns"
  }
}
```

The map requires `schema_version: 1`, `source_fingerprint`, `source_cache: {"path": "...", "sha256": "..."}`, and a `frames` array containing `frame_index`, `canonical_source_timestamp_ns` and `cam0_header_ns`. Cache paths resolve beside the map file. The adapter rehashes the actual schema-2 cache, checks its fingerprint, verifies each canonical/camera-0 pair against the cache's per-camera evidence, requires unique increasing indices and both clocks, and checks every requested canonical timestamp against GT. Output retains the original GT timestamp. The Python API takes `source_clock_map=path` with the exported `MAPPED_CLOCK_ALIGNMENT`; omitting the map keeps the existing exact-cam0 behavior. The light preflight uses this mapping for 73 of 300 reviewed samples, without discarding samples or treating them as unavailable.

Yaw is passed through **unchanged** for `orient_valid=1`; `orient_valid=0` yields null. The preferred producer's yaw was user-confirmed, and the adapter applies no 180-degree adjustment, zero substitution, or velocity-derived facing. It retains every published row, including the producer's inherent position/heading grace behavior; no additional confidence or age filter is applied. Calibration declarations are provenance assertions, not automatic physical verification. ROI cropping is applied afterward by the common evaluator.

## DeepStream DB3 input

Inspect an existing example without asserting alignment or computing scores:

```bash
python3 -B -m tracking_evaluation inspect-deepstream \
  /robodata/gdc_atrium/calib_gridwalk_deepstream/mv3dt_tracks/gridwalk_tracks_0.db3
```

The reader supports the existing ROS2 `vision_msgs/Detection3DArray` SQLite recording via the repository's pure CDR decoder, without a ROS installation, video processing, dataset constructors, or cache writes. It preserves the IDs recorded in the messages. Known gridwalk/close-gridwalk recordings provide cached asynchronous tracks, publish-time timestamps, and placeholder identity quaternions; their yaw is unavailable and is excluded from angular scoring.

The older `/robodata/gdc_atrium/ICRA_GDC/deepstream_fixed/dense_tracks_new.csv` is a repaired historical artifact, not the preferred scored CSV above. Its repair, gaps, conflicting records, and regeneration requirements remain documented in the [preparation report](/robodata/gdc_atrium/ICRA_GDC/evaluation_preparation_20260915/README.md). The independent regenerated MV3DT output is a separately labeled alternative baseline. For DB3 recordings, the [alignment template](examples/deepstream_alignment.template.json) remains intentionally unverified; fill in actual sequence identity, common-world calibration evidence and an affine time mapping:

`GT_relative_seconds = scale * ((header_ns - header_origin_ns) / 1e9) + offset_s`

Use `--prediction-format deepstream --alignment /path/to/verified_alignment.json`. When GT supplies a dataset ID, the alignment's `sequence_id` must match it. Nearest-snapshot selection is limited by explicit time tolerance and recorded coverage; it never clamps to endpoints. Default missing-correspondence behavior is an error. Explicit `absence_policy: "empty"` is available but documented per frame as a user-selected policy, not observed absence. Cache staleness remains unknown, and the report never claims these are fresh synchronized detections.

## Runtime and validation

Each report separates GT loading, prediction loading and metric computation, and records wall time, CPU time, peak process RSS, solver backend, and throughput. End-to-end wall time excludes writing the final report files. A run with already-cached filesystem pages is not a cold-storage benchmark.

```bash
python3 -B -m unittest discover -s tracking_evaluation/tests -v
python3 -B -m unittest discover -s tests -p test_deepstream_input.py -v
```

Tests cover one-to-one matching, arbitrary identity namespaces, identity switches and gaps, circular wrapping, count aggregation, HOTA thresholds, missing/empty frames, fingerprint and timestamp mismatches, explicit prediction policies, ROI boundaries, output preservation, and DeepStream parsing/alignment. CSV tests additionally cover exact integer timestamps, full-file rejection, source changes, explicit clock/calibration declarations, missing versus processed-empty ticks, raw yaw preservation, and CLI ledger/alignment paths. CLEAR tests cover continuity versus nearest/cardinality assignments, fragmentation, empty-frame state, aggregation, and gate boundaries.

Independent differential validation compared HOTA and IDF1 against official TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`: 132 sequence checks and four aggregate checks, across both similarities and both assignment backends. All 16,456 compared values agreed within `1.11e-16`. The [verification record](validation/trackeval_reference_verification.json) includes source hashes and versions; [reproduction instructions](validation/README.md) accompany the preserved harness. Undefined empty IDF1 is intentionally `null` here versus zero in TrackEval. Base MOTA/MOTP and circular yaw have separate analytical and exhaustive tests.

The new [CLEAR verification record](/robodata/gdc_atrium/ICRA_GDC/dense_pipeline_eval_20260915/clear_reference/verification.json) compares the separate CLEAR implementation against the same pinned official commit, including all four real dense variants, 128 random sequences and 15 named edge cases, with both assignment backends: **294 sequence comparisons and two aggregate comparisons**, all passed. TP/FP/FN/IDSW/Frag and MOTA agree exactly; maximum mean-similarity difference is `2.22e-16`. The preserved harness is [verify_clear_reference.py](validation/verify_clear_reference.py). Adding CLEAR leaves the existing HOTA, IDF1, base position, and circular heading calculations intact; the dense report includes regression and official-reference checks for those fields as well.
