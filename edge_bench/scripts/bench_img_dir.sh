#!/bin/bash
# Evaluate MNN accuracy on one or more flat image folders (mnn_bench --image-dir mode).
#
# Unlike bench_linux.sh (dataset JSON mode), ground truth comes from the file name,
# exactly as mnn_bench parses it:  <classID>_<latin name>_<index>.png  ->  classID.
# A folder that holds no .png itself but has rgb/ and/or rgb_with_bg/ sub-folders is
# expanded into those, which matches the layout of the generalization test sets.
#
# Paths may be given relative to the current directory or to edge_bench/ (the script
# cd's there, like bench_linux.sh), so both "edge_bench/data/x" and "data/x" work.
#
# Every (folder, model) pair runs in its own process (--model) so peak/mean RSS only
# cover that run. The per-run JSONs are then merged into a single file shaped like
# `mnn_bench --output-json`, with each model entry tagged by its "image_dir" plus a
# derived "accuracy_pct"; a summary table is printed and an optional CSV is written.
#
# Note: --image-dir mode cannot save per-image predictions (mnn_bench only fills
# y_true/y_pred for --dataset-json), so runs are compared by top1/accuracy only, and
# the per-image "[image] ..." lines mnn_bench prints are suppressed unless
# --keep-image-lines is given.
#
# Usage:
#   bash edge_bench/scripts/bench_img_dir.sh \
#     --img-dirs edge_bench/data/seed_cls_cell_phone_test \
#     --model-dir data/models/scratch/mnn/fp16 \
#     --max-images 200 --output-json edge_bench/output/img_dir_eval/phone.json
set -euo pipefail
CALL_PWD="$PWD"
cd "$(dirname "$0")/.."

BINARY="${BINARY:-build/mnn_bench}"
BUILD_DIR="${BUILD_DIR:-build}"
MODEL_DIR="${MODEL_DIR:-data/models_mnn}"
IMG_DIRS="${IMG_DIRS:-}"
BACKEND="${BACKEND:-CPU}"
THREADS="${THREADS:-4}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-1}"
MAX_IMAGES="${MAX_IMAGES:-0}"
SEED="${SEED:-42}"
# Deliberately a sub-directory: edge_bench/output/*.json is globbed by
# scripts/plot_mnn_figures.py as one latency run per device/backend file.
OUTPUT_JSON="${OUTPUT_JSON:-output/img_dir_eval/eval_$(date +%s).json}"
CSV_PATH="${CSV_PATH:-}"
KEEP_IMAGE_LINES="${KEEP_IMAGE_LINES:-false}"

FORCE=false
OVERWRITE=false
CMAKE_ARGS=""
EXTRA_ARGS=()

print_usage() {
    cat >&2 <<'USAGE'
Usage: bash edge_bench/scripts/bench_img_dir.sh --img-dirs DIR[,DIR...] [options]

  --img-dirs a,b,c      Image folders to evaluate (repeatable). Each folder must hold
                        the images directly; a folder with rgb/ and rgb_with_bg/
                        sub-folders is expanded into those.
  --img-dir DIR         Same as --img-dirs, one folder per flag (repeatable).
  --model-dir DIR       Directory of *.mnn models, scanned recursively (default: data/models_mnn)
  --backend NAME        CPU | OPENCL | CUDA | METAL (default: CPU)
  --threads N           (default: 4)
  --warmup N            Warm-up iterations on the first image (default: 10)
  --repeat N            Iterations for the latency-only path (default: 1)
  --max-images N        Random subset size per folder, 0 = all (default: 0)
  --seed N              Shuffle seed for --max-images (default: 42)
  --output-json PATH    Merged result file (default: output/img_dir_eval/eval_<ts>.json)
  --csv PATH            Also write the summary table as CSV
  --keep-image-lines    Do not hide mnn_bench's per-image "[image] ..." stdout lines
  --binary PATH         mnn_bench binary (default: build/mnn_bench)
  --build-dir DIR       CMake build dir (default: build)
  --cmake-args "..."    Extra cmake -D... arguments for the auto-build
  --force               Rebuild mnn_bench even if the binary exists
  --overwrite           Overwrite an existing --output-json

  Anything else is passed straight to mnn_bench (e.g. --precision high, --topk 5,
  --width 224, --no-progress).
USAGE
}

add_img_dir() {
    if [[ -z "$IMG_DIRS" ]]; then
        IMG_DIRS="$1"
    else
        IMG_DIRS="$IMG_DIRS,$1"
    fi
}

# Accept paths relative to edge_bench/ (bench_linux.sh convention) or to the directory
# the script was called from.
resolve_path() {
    if [[ -e "$1" ]]; then
        printf '%s' "$1"
    elif [[ -e "$CALL_PWD/$1" ]]; then
        printf '%s' "$CALL_PWD/$1"
    else
        return 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --img-dirs | --img-dir)
        add_img_dir "$2"
        shift 2
        ;;
    --model-dir)
        MODEL_DIR="$2"
        shift 2
        ;;
    --backend)
        BACKEND="$2"
        shift 2
        ;;
    --threads)
        THREADS="$2"
        shift 2
        ;;
    --warmup)
        WARMUP="$2"
        shift 2
        ;;
    --repeat)
        REPEAT="$2"
        shift 2
        ;;
    --max-images)
        MAX_IMAGES="$2"
        shift 2
        ;;
    --seed)
        SEED="$2"
        shift 2
        ;;
    --output-json)
        OUTPUT_JSON="$2"
        shift 2
        ;;
    --csv)
        CSV_PATH="$2"
        shift 2
        ;;
    --binary)
        BINARY="$2"
        shift 2
        ;;
    --build-dir)
        BUILD_DIR="$2"
        shift 2
        ;;
    --keep-image-lines)
        KEEP_IMAGE_LINES=true
        shift
        ;;
    --cmake-args)
        CMAKE_ARGS="$2"
        shift 2
        ;;
    --force)
        FORCE=true
        shift
        ;;
    --overwrite)
        OVERWRITE=true
        shift
        ;;
    -h | --help)
        print_usage
        exit 0
        ;;
    *)
        EXTRA_ARGS+=("$1")
        shift
        ;;
    esac
done

if [[ -z "$IMG_DIRS" ]]; then
    echo "ERROR: at least one image folder is required (--img-dirs a,b or --img-dir a)." >&2
    print_usage
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve image folders. IMG_DIR_ARR holds the paths handed to mnn_bench,
# IMG_LABEL_ARR the names used in the merged JSON / table / CSV.
# ---------------------------------------------------------------------------
IMG_DIR_ARR=()
IMG_LABEL_ARR=()
IFS=',' read -r -a RAW_DIRS <<< "$IMG_DIRS"
for RAW_DIR in "${RAW_DIRS[@]}"; do
    LABEL="${RAW_DIR%/}"
    [[ -z "$LABEL" ]] && continue
    if ! DIR="$(resolve_path "$LABEL")"; then
        echo "ERROR: image folder not found: $LABEL" >&2
        echo "       (tried relative to edge_bench/ and to $CALL_PWD)" >&2
        exit 1
    fi
    if compgen -G "$DIR/*.png" > /dev/null; then
        IMG_DIR_ARR+=("$DIR")
        IMG_LABEL_ARR+=("$LABEL")
        continue
    fi
    # Parent of a rgb/ + rgb_with_bg/ pair (generalization test-set layout)?
    EXPANDED=false
    for SUB in rgb rgb_with_bg; do
        if [[ -d "$DIR/$SUB" ]] && compgen -G "$DIR/$SUB/*.png" > /dev/null; then
            IMG_DIR_ARR+=("$DIR/$SUB")
            IMG_LABEL_ARR+=("$LABEL/$SUB")
            EXPANDED=true
        fi
    done
    if [[ "$EXPANDED" != "true" ]]; then
        echo "ERROR: no .png in $LABEL (nor in $LABEL/rgb, $LABEL/rgb_with_bg)." >&2
        echo "       --image-dir mode reads all .png of one folder, pass the folder that holds them." >&2
        exit 1
    fi
done

if [[ ${#IMG_DIR_ARR[@]} -eq 0 ]]; then
    echo "ERROR: no usable image folder given." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required to merge per-run JSON results." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build (only if needed).
# ---------------------------------------------------------------------------
if [[ "$FORCE" == "true" || ! -x "$BINARY" ]]; then
    echo "Building mnn_bench..."
    CMAKE_ARG_ARR=()
    if [[ -n "$CMAKE_ARGS" ]]; then
        read -r -a CMAKE_ARG_ARR <<< "$CMAKE_ARGS"
    fi
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "${CMAKE_ARG_ARR[@]}"
    cmake --build "$BUILD_DIR" -j16
fi
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binary not found: $BINARY" >&2
    exit 1
fi

if [[ -f "$OUTPUT_JSON" && "$OVERWRITE" != "true" ]]; then
    echo "Skip: output exists ($OUTPUT_JSON); use --overwrite to overwrite."
    exit 0
fi

# ---------------------------------------------------------------------------
# Models.
# ---------------------------------------------------------------------------
if RESOLVED="$(resolve_path "$MODEL_DIR")"; then
    MODEL_DIR="$RESOLVED"
fi

MODELS=()
while IFS= read -r m; do
    MODELS+=("$m")
done < <(find "$MODEL_DIR" -type f -name '*.mnn' | sort)

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: no .mnn models found in: $MODEL_DIR" >&2
    exit 1
fi

TOTAL_RUNS=$(( ${#IMG_DIR_ARR[@]} * ${#MODELS[@]} ))
echo "Image folders (${#IMG_DIR_ARR[@]} x ${#MODELS[@]} models = $TOTAL_RUNS runs):"
for i in "${!IMG_DIR_ARR[@]}"; do
    DIR="${IMG_DIR_ARR[$i]}"
    PNG_COUNT=$(find "$DIR" -maxdepth 1 -type f -name '*.png' | wc -l)
    # mnn_bench keeps a .png only when its name starts with "<classID>_".
    UNLABELLED=0
    while IFS= read -r IMG; do
        NAME="$(basename "$IMG")"
        [[ "$NAME" =~ ^[0-9]+_ ]] || UNLABELLED=$((UNLABELLED + 1))
    done < <(find "$DIR" -maxdepth 1 -type f -name '*.png')
    printf '  %s (%s images' "${IMG_LABEL_ARR[$i]}" "$PNG_COUNT"
    if [[ "$UNLABELLED" -gt 0 ]]; then
        printf ', %s skipped: no <classID>_ prefix' "$UNLABELLED"
    fi
    printf ')\n'
done

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# ---------------------------------------------------------------------------
# Run every (folder, model) pair.
# ---------------------------------------------------------------------------
run_mnn() {
    if [[ "$KEEP_IMAGE_LINES" == "true" ]]; then
        "$BINARY" "$@"
    else
        # --image-dir mode prints one "[image] ..." line per picture; keep the summary.
        "$BINARY" "$@" | grep -v '^\[image\] '
    fi
}

PARTS=()
FAILED_RUNS=()
INDEX=0
for i in "${!IMG_DIR_ARR[@]}"; do
    DIR="${IMG_DIR_ARR[$i]}"
    LABEL="${IMG_LABEL_ARR[$i]}"
    for MODEL in "${MODELS[@]}"; do
        INDEX=$((INDEX + 1))
        TEMP_JSON="$TMP_DIR/run_$(printf '%04d' "$INDEX").json"
        echo "[$INDEX/$TOTAL_RUNS] $(basename "$MODEL") @ $LABEL"
        RUN_ARGS=(--model "$MODEL" --backend "$BACKEND" --threads "$THREADS" \
                  --warmup "$WARMUP" --repeat "$REPEAT" --image-dir "$DIR" \
                  --max-images "$MAX_IMAGES" --seed "$SEED" --output-json "$TEMP_JSON")
        if ! run_mnn "${RUN_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
            echo "  ERROR: run failed: $MODEL @ $LABEL" >&2
            FAILED_RUNS+=("$MODEL @ $LABEL")
            continue
        fi
        # Tag the run with its folder (mnn_bench only records the model path) and
        # report its top1 right away, since the run itself is otherwise silent.
        ACC_LINE=$(python3 - "$TEMP_JSON" "$LABEL" <<'PY'
import json, sys

path, image_dir = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
for entry in data.get("models", []):
    entry["image_dir"] = image_dir
with open(path, "w") as f:
    json.dump(data, f, indent=2)

entry = data["models"][0]
total = int(entry.get("total") or 0)
print(f"{entry['top1']}/{total} = {entry['top1'] / total * 100:.2f}%" if total else "no images")
PY
)
        echo "  top1: $ACC_LINE"
        PARTS+=("$TEMP_JSON")
    done
done

if [[ ${#PARTS[@]} -eq 0 ]]; then
    echo "ERROR: every run failed; nothing to write to $OUTPUT_JSON" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Merge per-run JSON into one mnn_bench-shaped file + print/collect a table.
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$OUTPUT_JSON")"
IMG_LABELS_RESOLVED="$(printf '%s\n' "${IMG_LABEL_ARR[@]}")" \
    python3 - "$OUTPUT_JSON" "$CSV_PATH" "${PARTS[@]}" <<'PY'
import csv
import json
import os
import sys

out_path = sys.argv[1]
csv_path = sys.argv[2]
part_paths = sys.argv[3:]
image_dirs = [d for d in os.environ.get("IMG_LABELS_RESOLVED", "").splitlines() if d]

merged = None
models = []
for part_path in part_paths:
    with open(part_path) as f:
        run = json.load(f)
    if merged is None:
        # Same top-level shape that `mnn_bench --output-json` writes for one run.
        merged = {k: v for k, v in run.items() if k != "models"}
    for entry in run.get("models", []):
        total = int(entry.get("total") or 0)
        entry["accuracy_pct"] = round(entry["top1"] / total * 100.0, 2) if total else None
        models.append(entry)

if merged is None:
    merged = {}
merged["image_dirs"] = image_dirs
merged["models"] = models

with open(out_path, "w") as f:
    json.dump(merged, f, indent=2, sort_keys=True)
    f.write("\n")

rows = sorted(models, key=lambda e: (e["image_dir"], e["model"]))
header = ("image_dir", "model", "top1/total", "accuracy (%)", "latency (ms)", "max RSS (MB)")
table = [
    (
        e["image_dir"],
        e["model"],
        f"{e['top1']}/{e['total']}",
        "-" if e["accuracy_pct"] is None else f"{e['accuracy_pct']:.2f}",
        f"{e['latency_ms_mean']:.2f}",
        f"{e['max_rss_kb'] / 1024.0:.1f}",
    )
    for e in rows
]
widths = [max(len(header[i]), *(len(r[i]) for r in table)) if table else len(header[i]) for i in range(len(header))]
print(f"Merged {len(part_paths)} run(s) into {out_path}")
print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
for r in table:
    print("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))

if csv_path:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fields = ["image_dir", "model", "quant", "top1", "top3", "total", "accuracy_pct",
              "latency_ms_mean", "mean_rss_kb", "max_rss_kb", "file_size_bytes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for e in rows:
            writer.writerow({**e, "quant": os.path.basename(os.path.dirname(e["model"]))})
    print(f"CSV: {csv_path}")
PY

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo "WARNING: ${#FAILED_RUNS[@]} run(s) failed:" >&2
    printf '  %s\n' "${FAILED_RUNS[@]}" >&2
fi
echo "Result: $OUTPUT_JSON"
