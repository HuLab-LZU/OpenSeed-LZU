# MNN Edge Benchmark

Cross-platform inference benchmark for the 14 MNN models (seed 42) on
FP32 / FP16 / INT8, across CPU / OpenCL / CUDA / Metal.

## Structure

```text
edge_bench/
├── CMakeLists.txt
├── tools/mnn_bench.cc
├── config/
│   ├── models_14.txt
│   └── devices.json
├── scripts/
│   ├── convert_mnn.py
│   └── run_benchmarks.py
└── output/
```

## Steps

### 1. Dataset

`mnn_bench` reads the dataset directly with `MNN::CV::imread` from an
`OpenSeed-LZU`-like directory (contains `rgb/`, `rgb_with_bg/` and
`cls_<tag>_<modality>_<task>_<split_type>_<split>.json`):

```bash
eedge_bench/build/mnn_bench \
  --model edge_bench/output/mnn/fp16/resnet_50.mnn \
  --dataset-json data/datasets/OpenSeed-LZU/cls_656_30_rgb_st_crop_test.json \
  --base-dirs rgb,rgb_with_bg --max-images 1000
```

If `--dataset-dir` is omitted, latency/memory-only mode is used.


### 2. Convert ONNX -> MNN FP32 / FP16 / INT8

```bash
uv run python edge_bench/scripts/convert_mnn.py \
  --onnx-dir /path/to/onnx_models \
  --out-dir edge_bench/output/mnn \
  --calibration-dir edge_bench/data/calib_656/images

# 如果 Python mnnquant 不可用，可指定 C++ quantized.out
uv run python edge_bench/scripts/convert_mnn.py \
  --onnx-dir /path/to/onnx_models \
  --out-dir edge_bench/output/mnn \
  --calibration-dir edge_bench/data/calib_656/images \
  --mnnquant /path/to/build/quantized.out
```

The script scans `--onnx-dir` and converts every `*.onnx` file it finds, so
no model list file is needed. The script scans every `*.onnx` directly; for
`<base>.onnx` or `<base>_f16.onnx` it writes `<out>/<prec>/<base>.mnn`.
No `*_42` suffix convention is required.
If a model is missing or a precision cannot be converted, it is skipped with a
warning.

### 3. Build mnn_bench

MNN is fetched automatically via FetchContent and MNN backends are
auto-detected based on the compile platform:

- macOS / Apple -> `MNN_METAL=ON`
- Linux/Windows with CUDA toolkit found -> `MNN_CUDA=ON`
- Linux/Android with OpenCL found -> `MNN_OPENCL=ON`
- Vulkan stays off unless explicitly enabled

You can override any of them with `-DMNN_*=ON/OFF` (e.g. `-DMNN_OPENCL=OFF`).

The default MNN git tag is `3.2.0` (no `v` prefix). Override with:
```bash
cmake -S edge_bench -B edge_bench/build -DMNN_TAG=3.2.0
```

> CUDA 13 note: when CUDA is enabled, `edge_bench/CMakeLists.txt` sets
> `MNN_CUDA_NATIVE_ARCH=ON`, so MNN only compiles for the local GPU arch
> (e.g. `sm_89`) and skips the obsolete `compute_60/61/...` targets that
> CUDA 13 no longer supports. A reference patch is kept at
> `patches/mnn_cuda_arch.patch` if you prefer to patch the broad gencode list
> instead.
>
> `patches/mnn_cuda_softmax_fp16.patch` is applied automatically (via
> `cmake/apply_patch.cmake`) and is **not** optional. MNN 3.6.1's CUDA Softmax
> execution detects fp16 from the tensor type instead of the backend precision;
> because the CUDA backend keeps float tensors declared as fp32 and only halves the
> buffer, the fp32 kernel reads an fp16 buffer and every Softmax output becomes NaN.
> Without the patch, `--precision low` on CUDA reports ~0.5% accuracy (all-NaN
> logits) for every attention model: ViT, Swin, MobileViT and the MobileNetV4
> hybrids. CPU / OpenCL / Metal and `--precision normal|high` are unaffected.

#### Build examples

```bash
# CPU-only / auto-detected (default)
cmake -S edge_bench -B edge_bench/build
cmake --build edge_bench/build -j

# Explicitly enable OpenCL + CUDA (e.g. on a Linux x86 machine with both available)
cmake -S edge_bench -B edge_bench/build \
  -DMNN_OPENCL=ON -DMNN_CUDA=ON

# Explicitly enable Metal (macOS)
cmake -S edge_bench -B edge_bench/build -DMNN_METAL=ON
```

### 4. Run on one device manually

```bash
edge_bench/build/mnn_bench \
  --model edge_bench/output/mnn/fp16/resnet_50.mnn \
  --backend OPENCL --threads 4 \
  --dataset-json data/datasets/OpenSeed-LZU/cls_656_30_rgb_st_crop_test.json \
  --base-dirs rgb,rgb_with_bg --max-images 1000 \
  --output-json edge_bench/output/result.json
```

- MNN is built with `MNN_BUILD_OPENCV=ON`, `MNN_IMGCODECS=ON`, and the
  benchmark target defines `MNN_USE_OPENCV`.

### 4b. Debug example images

Load images from a folder whose filenames follow
`<classID>_<latinName>_<number>.png` and print per-image predictions:

```bash
eedge_bench/build/mnn_bench \
  --model edge_bench/output/mnn/fp16/resnet_50.mnn \
  --example-dir edge_bench/data/example_imgs \
  --topk 5 --warmup 2 --repeat 1
```

Output example:
```text
[example] 106_Oxybasis glauca_5.png | true=106 (Oxybasis glauca) | pred=588,492,469,50,272
```

### 5. Run all devices

Configure `edge_bench/config/devices.json` with local / ssh / adb entries,
then:

```bash
uv run python edge_bench/scripts/run_benchmarks.py \
  --devices edge_bench/config/devices.json \
  --mnn-dir edge_bench/output/mnn \
  --output edge_bench/output/benchmark_results.csv
```

For remote devices, make sure the `mnn_bench` binary, MNN model files and
prepared data have already been synced to the paths in `devices.json`.

## Notes

- Images are read with `MNN::CV::imread`, then preprocessed with
  `MNN::CV::resize` (resize to 224x224 + seeds_rgb normalization), and
  converted to NC4HW4 before `Module::onForward`.
- If `--dataset-dir` is omitted, accuracy metrics are skipped.
- Memory: `maxrss` is normalized to KB (Linux/Android/macOS). On macOS,
  `ru_maxrss` is reported in bytes (unlike Linux, where it is already in KB),
  so it is divided by 1024.
- `mnn_bench` benchmarks a **single** model per invocation (`--model`). The
  platform scripts iterate over every `.mnn` in the model directory, running one
  `--model` invocation per process. Because each model runs in its own process,
  the sampled RSS and the peak RSS (`getrusage` high-water mark) reflect only
  that model, instead of accumulating across every model loaded into one
  long-lived process. The scripts accept `--model-dir <dir>` (or the `MODEL_DIR`
  env var) to choose which folder of `.mnn` models to benchmark; it is consumed
  by the script and is **not** a `mnn_bench` option.

## Multi-task / species-only note

`mnn_bench` uses the **first output** of the MNN model for top-1/top-3 accuracy
and latency. For multi-task models exported with `--mode multi`, the first
output is `species_output`. The genus and family heads are auxiliary during
training and are **not** benchmarked on edge devices in the current study. See
`docs/mnn_edge_decision.md` for details.

## Platform scripts

Build and benchmark helpers under `edge_bench/scripts/`.

### Linux / macOS

```bash
# Benchmark; compiles automatically if the binary is missing
bash edge_bench/scripts/bench_linux.sh --backend CPU --threads 4
bash edge_bench/scripts/bench_macos.sh --backend METAL

# Force recompile / overwrite existing result
bash edge_bench/scripts/bench_linux.sh --force --overwrite
```

### Android (NDK + adb)

```bash
# Set ANDROID_NDK_HOME; binary is cross-compiled automatically if missing
bash edge_bench/scripts/bench_android.sh --backend CPU --threads 4
bash edge_bench/scripts/bench_android.sh --force --overwrite
```

### Windows (PowerShell)

```powershell
.\scripts\bench_windows.ps1 -Backend CPU -Threads 4
.\scripts\bench_windows.ps1 -Backend CPU -Threads 4 -Force -Overwrite
```

All benchmark scripts discover every `.mnn` model under the model directory,
run one `mnn_bench --model` per process, and merge the per-model JSON results
into one file with the same shape that `mnn_bench --output-json` writes
(`backend`/`threads`/`warmup`/`repeat`/`max_images`/`models`). Models that
fail are reported as warnings and skipped.
