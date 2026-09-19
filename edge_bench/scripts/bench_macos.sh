#!/bin/bash
# Build (if needed) and benchmark mnn_bench on macOS.
#
# Each model is benchmarked in its own process (--model) so that peak/max/mean
# RSS reflect only that model. The per-model JSON results are then merged into
# a single file with the same structure that `mnn_bench --output-json` writes
# (backend/threads/warmup/repeat/max_images/models).
set -euo pipefail
cd "$(dirname "$0")/.."

BINARY="${BINARY:-build/mnn_bench}"
BUILD_DIR="${BUILD_DIR:-build}"
MODEL_DIR="${MODEL_DIR:-data/models_mnn}"
BACKEND="${BACKEND:-CPU}"
THREADS="${THREADS:-4}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-1000}"
DATASET_JSON="${DATASET_JSON:-}"
BASE_DIRS="${BASE_DIRS:-rgb,rgb_with_bg}"
MAX_IMAGES="${MAX_IMAGES:-1000}"
OUTPUT_JSON="${OUTPUT_JSON:-output/result_$(date +%s).json}"

FORCE=false
OVERWRITE=false
CMAKE_ARGS=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
  --backend)
    BACKEND="$2"
    shift 2
    ;;
  --model-dir)
    MODEL_DIR="$2"
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
  --output-json)
    OUTPUT_JSON="$2"
    shift 2
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
  *)
    EXTRA_ARGS+=("$1")
    shift
    ;;
  esac
done

if [[ "$FORCE" == "true" || ! -x "$BINARY" ]]; then
  echo "Building mnn_bench..."
  CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
  CMAKE_ARG_ARR=()
  if [[ -n "$CMAKE_ARGS" ]]; then
    read -r -a CMAKE_ARG_ARR <<<"$CMAKE_ARGS"
  fi
  cmake -S . -B "$BUILD_DIR" -G Ninja ${CMAKE_ARG_ARR[@]+"${CMAKE_ARG_ARR[@]}"}
  cmake --build "$BUILD_DIR" -j"$CORES"
fi
if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: binary not found: $BINARY" >&2
  exit 1
fi

if [[ -f "$OUTPUT_JSON" && "$OVERWRITE" != "true" ]]; then
  echo "Skip: output exists ($OUTPUT_JSON); use --overwrite to overwrite."
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to merge per-model JSON results." >&2
  exit 1
fi

# Discover models (bash 3.2 compatible; macOS ships bash 3.2).
MODELS=()
while IFS= read -r m; do
  MODELS+=("$m")
done < <(find "$MODEL_DIR" -type f -name '*.mnn' | sort)

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "ERROR: no .mnn models found in: $MODEL_DIR" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PARTS=()
FAILED_MODELS=()
INDEX=0
for MODEL in "${MODELS[@]}"; do
  INDEX=$((INDEX + 1))
  TEMP_JSON="$TMP_DIR/model_$(printf '%03d' "$INDEX").json"
  RUN_ARGS=(--model "$MODEL" --backend "$BACKEND" --threads "$THREADS"
    --warmup "$WARMUP" --repeat "$REPEAT")
  if [[ -n "$DATASET_JSON" ]]; then
    RUN_ARGS+=(--dataset-json "$DATASET_JSON" --base-dirs "$BASE_DIRS"
      --max-images "$MAX_IMAGES")
  fi
  RUN_ARGS+=(--output-json "$TEMP_JSON")
  echo "[$INDEX/${#MODELS[@]}] $(basename "$MODEL")"
  if ! "$BINARY" "${RUN_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
    echo "  ERROR: model failed: $MODEL" >&2
    FAILED_MODELS+=("$MODEL")
    continue
  fi
  PARTS+=("$TEMP_JSON")
done

if [[ ${#PARTS[@]} -eq 0 ]]; then
  echo "ERROR: all models failed; nothing to write to $OUTPUT_JSON" >&2
  exit 1
fi

# Merge per-model JSON into a single file, keeping the mnn_bench output shape.
mkdir -p "$(dirname "$OUTPUT_JSON")"
python3 - "$OUTPUT_JSON" "${PARTS[@]}" <<'PY'
import json, sys

out_path = sys.argv[1]
part_paths = sys.argv[2:]

merged = None
models = []
for p in part_paths:
    with open(p) as f:
        d = json.load(f)
    if merged is None:
        merged = {k: d[k] for k in ("backend", "threads", "warmup", "repeat", "max_images")}
    models.extend(d.get("models", []))

if merged is None:
    merged = {}
merged["models"] = models

with open(out_path, "w") as f:
    json.dump(merged, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"Merged {len(part_paths)} model result(s) into {out_path}")
PY

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  echo "WARNING: ${#FAILED_MODELS[@]} model(s) failed:" >&2
  printf '  %s\n' "${FAILED_MODELS[@]}" >&2
fi
echo "Result: $OUTPUT_JSON"
