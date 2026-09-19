#!/usr/bin/env python3
"""Convert ONNX models to MNN FP32 / FP16 / INT8.

The script scans ``--onnx-dir`` for every ``*.onnx`` file and converts each
base name directly (no ``*_42`` convention). For each ONNX file named:

  <onnx-dir>/<base>.onnx
  <onnx-dir>/<base>_f16.onnx

it writes:

  <out>/fp32/<base>.mnn
  <out>/fp16/<base>.mnn
  <out>/int8/<base>.mnn

If several ONNX files map to the same base name, ``<base>.onnx`` is preferred
over ``<base>_f16.onnx`` (the fp16 ONNX is treated as a derived variant).

INT8 uses the MNN ``mnnquant`` tool with a calibration image list. If a
precision cannot be converted (for example INT8 calibration data is missing),
that model is skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# The script scans every ``*.onnx`` file directly (no ``*_42`` convention).
# Generated fp16 ONNX variants (``*_f16.onnx``) are de-duplicated back to the
# same base name so they are not treated as separate models.
ONNX_NAME_SUFFIXES = ("_f16.onnx", ".onnx")


def model_base_from_onnx(path: Path) -> str:
    name = path.name
    for suffix in ONNX_NAME_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Unsupported ONNX filename: {name}")


def onnx_priority(path: Path) -> int:
    """Lower value wins when multiple files map to the same base name."""
    if path.name.endswith("_f16.onnx"):
        return 1
    return 0


def discover_models(onnx_dir: Path) -> dict[str, Path]:
    """Return ``{base: onnx_path}`` for every ``*.onnx`` in ``onnx_dir``."""
    models: dict[str, Path] = {}
    if not onnx_dir.is_dir():
        print(f"error: ONNX directory not found: {onnx_dir}", file=sys.stderr)
        return models
    for path in sorted(onnx_dir.glob("*.onnx")):
        base = model_base_from_onnx(path)
        current = models.get(base)
        if current is None or onnx_priority(path) < onnx_priority(current):
            models[base] = path
    return models


def run_cmd(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def convert(args: argparse.Namespace) -> int:
    models = discover_models(args.onnx_dir)
    if not models:
        print(f"no ONNX models found in {args.onnx_dir}", file=sys.stderr)
        return 0

    for base, onnx in models.items():
        for prec in ["fp32", "fp16", "int8"]:
            flag = getattr(args, prec)
            if not flag:
                continue
            out_dir = args.out_dir / prec
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{base}.mnn"

            if out_path.exists() and not args.force:
                print(f"skip existing {prec}: {out_path}")
                continue

            if prec == "fp32":
                run_cmd([
                    "mnnconvert",
                    "-f",
                    "ONNX",
                    "--modelFile",
                    str(onnx),
                    "--bizCode",
                    f"{base}@OpenSeed",
                    "--MNNModel",
                    str(out_path),
                ])
            elif prec == "fp16":
                run_cmd([
                    "mnnconvert",
                    "-f",
                    "ONNX",
                    "--modelFile",
                    str(onnx),
                    "--bizCode",
                    f"{base}@OpenSeed",
                    "--fp16",
                    "--MNNModel",
                    str(out_path),
                ])
            elif prec == "int8":
                fp32_path = args.out_dir / "fp32" / f"{base}.mnn"
                if not fp32_path.exists():
                    run_cmd([
                        "mnnconvert",
                        "-f",
                        "ONNX",
                        "--modelFile",
                        str(onnx),
                        "--MNNModel",
                        str(fp32_path),
                    ])
                if not args.calibration_dir.is_dir():
                    print(
                        f"skip INT8 for {base}: calibration dir missing ({args.calibration_dir})",
                        file=sys.stderr,
                    )
                    continue
                calib_images = (
                    list(args.calibration_dir.glob("*.png"))
                    + list(args.calibration_dir.glob("*.jpg"))
                    + list(args.calibration_dir.glob("*.jpeg"))
                )
                # Calibration images are uint8 0-255; MNN quant applies (src - mean) * normal.
                mean = [0.34865115 * 255.0, 0.29936219 * 255.0, 0.25752143 * 255.0]
                normal = [
                    1.0 / (0.2302609 * 255.0),
                    1.0 / (0.19996711 * 255.0),
                    1.0 / (0.17874425 * 255.0),
                ]
                config = {
                    "format": "RGB",
                    "mean": mean,
                    "normal": normal,
                    "width": 224,
                    "height": 224,
                    "path": str(args.calibration_dir),
                    "used_image_num": len(calib_images),
                    "feature_quantize_method": "KL",
                    "weight_quantize_method": "MAX_ABS",
                    "model": str(fp32_path),
                }
                config_path = args.out_dir / "int8" / f"{base}_config.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
                if args.mnnquant is not None:
                    if not args.mnnquant.exists():
                        print(
                            f"[warn] --mnnquant not found: {args.mnnquant}; skipping INT8 for {base}",
                            file=sys.stderr,
                        )
                        continue
                    try:
                        run_cmd([
                            str(args.mnnquant),
                            str(fp32_path),
                            str(out_path),
                            str(config_path),
                        ])
                    except subprocess.CalledProcessError as e:
                        print(
                            f"[warn] INT8 quantization failed for {base}: {e}",
                            file=sys.stderr,
                        )
                        continue
                else:
                    try:
                        run_cmd([
                            "mnnquant",
                            str(fp32_path),
                            str(out_path),
                            str(config_path),
                        ])
                    except subprocess.CalledProcessError as e:
                        print(
                            f"[warn] INT8 quantization failed for {base}: {e}",
                            file=sys.stderr,
                        )
                        continue
            print(f"done {base} {prec} -> {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("edge_bench/output/mnn"))
    p.add_argument("--calibration-dir", type=Path, default=Path("edge_bench/data/calib_656/images"))
    p.add_argument(
        "--mnnquant",
        type=Path,
        default=None,
        help="Path to C++ quantized.out binary (overrides Python mnnquant)",
    )
    p.add_argument("--fp32", action="store_true", default=False)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--int8", action="store_true", default=False)
    p.add_argument("--force", action="store_true", help="overwrite existing MNN models")
    args = p.parse_args()
    return convert(args)


if __name__ == "__main__":
    sys.exit(main())
