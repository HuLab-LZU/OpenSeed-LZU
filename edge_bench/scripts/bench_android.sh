#!/bin/bash
# Build (if needed) and benchmark mnn_bench on an Android device via adb.
set -euo pipefail
cd "$(dirname "$0")/.."

BINARY_LOCAL="${BINARY_LOCAL:-build-android/mnn_bench}"
MODEL_DIR="${MODEL_DIR:-data/models_mnn}"
BUILD_DIR="${BUILD_DIR:-build-android}"
NDK="${ANDROID_NDK_HOME:-}"
ABI="${ABI:-arm64-v8a}"
API="${API:-24}"
SERIAL="${SERIAL:-}"
REMOTE_BINARY="${REMOTE_BINARY:-/data/local/tmp/mnn_bench}"
REMOTE_LIB_DIR="${REMOTE_LIB_DIR:-/data/local/tmp/lib}"
REMOTE_MODEL_DIR="${REMOTE_MODEL_DIR:-/sdcard/mnn_models}"
REMOTE_DATASET_JSON="${REMOTE_DATASET_JSON:-}"
BACKEND="${BACKEND:-CPU}"
THREADS="${THREADS:-4}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-100}"
BASE_DIRS="${BASE_DIRS:-rgb,rgb_with_bg}"
MAX_IMAGES="${MAX_IMAGES:-1000}"
REMOTE_RESULT="${REMOTE_RESULT:-/sdcard/mnn_result.json}"
OUTPUT_JSON="${OUTPUT_JSON:-output/android_result_$(date +%s).json}"
PUSH_MODELS="${PUSH_MODELS:-false}"

FORCE=false
OVERWRITE=false
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
    --ndk)
        NDK="$2"
        shift 2
        ;;
    --serial)
        SERIAL="$2"
        shift 2
        ;;
    --push-models)
        PUSH_MODELS=true
        shift
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

if [[ -z "$SERIAL" ]]; then
    echo "ERROR: --s SERIAL or SERIAL env is required." >&2
    exit 1
fi

echo "Using Android device: $SERIAL"

if ! command -v adb >/dev/null 2>&1; then
    echo "ERROR: adb not found in PATH." >&2
    exit 1
fi

if [[ "$FORCE" == "true" || ! -x "$BINARY_LOCAL" ]]; then
    echo "Cross-compiling mnn_bench for Android..."
    if [[ -z "$NDK" ]]; then
        echo "ERROR: --ndk or ANDROID_NDK_HOME is required for Android build." >&2
        exit 1
    fi
    TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"
    if [[ ! -f "$TOOLCHAIN" ]]; then
        echo "ERROR: Android toolchain not found: $TOOLCHAIN" >&2
        exit 1
    fi
    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DANDROID_ABI="$ABI" \
        -DANDROID_PLATFORM="android-$API" \
        -DMNN_METAL=OFF -DMNN_CUDA=OFF -DMNN_OPENCL=ON -DMNN_VULKAN=OFF \
        -DMNN_BUILD_FOR_ANDROID_COMMAND=ON
    cmake --build "$BUILD_DIR" -j16
fi
if [[ ! -x "$BINARY_LOCAL" ]]; then
    echo "ERROR: binary not found: $BINARY_LOCAL" >&2
    exit 1
fi

if [[ -f "$OUTPUT_JSON" && "$OVERWRITE" != "true" ]]; then
    echo "Skip: output exists ($OUTPUT_JSON); use --overwrite to overwrite."
    exit 0
fi

adb -s "$SERIAL" shell mkdir -p "$REMOTE_LIB_DIR" >/dev/null 2>&1 || true
adb -s "$SERIAL" push "$BINARY_LOCAL" "$REMOTE_BINARY"
# Push all shared libraries produced by the Android build.
while IFS= read -r lib_path; do
    adb -s "$SERIAL" push "$lib_path" "$REMOTE_LIB_DIR/"
done < <(find "$BUILD_DIR" -name '*.so' -type f)

if [[ "$PUSH_MODELS" == "true" ]]; then
    adb -s "$SERIAL" push "$MODEL_DIR"/. "$REMOTE_MODEL_DIR"/
fi

adb -s "$SERIAL" shell chmod +x "$REMOTE_BINARY"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required to merge per-model JSON results." >&2
    exit 1
fi

# Discover the remote model list.
REMOTE_MODELS=()
while IFS= read -r m; do
    [[ -n "$m" ]] && REMOTE_MODELS+=("$m")
done < <(adb -s "$SERIAL" shell "find $REMOTE_MODEL_DIR -type f -name '*.mnn' | sort" | tr -d '\r')

if [[ ${#REMOTE_MODELS[@]} -eq 0 ]]; then
    echo "ERROR: no .mnn models found on device: $REMOTE_MODEL_DIR" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PARTS=()
FAILED_MODELS=()
INDEX=0
for REMOTE_MODEL in "${REMOTE_MODELS[@]}"; do
    INDEX=$((INDEX + 1))
    REMOTE_JSON="$REMOTE_RESULT.$INDEX"
    LOCAL_JSON="$TMP_DIR/model_$(printf '%03d' "$INDEX").json"
    REMOTE_ARGS=(--model "$REMOTE_MODEL" --backend "$BACKEND" --threads "$THREADS" \
                 --warmup "$WARMUP" --repeat "$REPEAT")
    if [[ -n "$REMOTE_DATASET_JSON" ]]; then
        REMOTE_ARGS+=(--dataset-json "$REMOTE_DATASET_JSON" --base-dirs "$BASE_DIRS" \
                       --max-images "$MAX_IMAGES")
    fi
    REMOTE_ARGS+=(--output-json "$REMOTE_JSON")
    REMOTE_CMD="LD_LIBRARY_PATH=$REMOTE_LIB_DIR $REMOTE_BINARY"
    for arg in "${REMOTE_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
        REMOTE_CMD+=" $(printf '%q' "$arg")"
    done
    echo "[$INDEX/${#REMOTE_MODELS[@]}] $(basename "$REMOTE_MODEL")"
    if ! adb -s "$SERIAL" shell "$REMOTE_CMD" >/dev/null 2>&1; then
        echo "  ERROR: model failed: $REMOTE_MODEL" >&2
        FAILED_MODELS+=("$REMOTE_MODEL")
        continue
    fi
    adb -s "$SERIAL" pull "$REMOTE_JSON" "$LOCAL_JSON" >/dev/null 2>&1
    adb -s "$SERIAL" shell rm -f "$REMOTE_JSON" >/dev/null 2>&1 || true
    if [[ -s "$LOCAL_JSON" ]]; then
        PARTS+=("$LOCAL_JSON")
    else
        echo "  ERROR: empty result for: $REMOTE_MODEL" >&2
        FAILED_MODELS+=("$REMOTE_MODEL")
    fi
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
