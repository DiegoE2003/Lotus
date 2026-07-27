#!/usr/bin/env python3
"""Convert DiLiGenT-Π Lotus normal .npy preds under 1-10/ to uint8 PNGs in 1-10_png/.

Expected layout (same as eval_diligentpi / DiligentPi-Outputs):

  /data/DiligentPi-Outputs/1-10/{d,g}/{OBJECT}/001.npy … 010.npy

Writes:

  /data/DiligentPi-Outputs/1-10_png/{d,g}/{OBJECT}/001.png … 010.png

NPYs are float [0, 1] HWC (infer.py / eval_diligentpi save_pred_like_infer).
PNGs match that vis path: (arr * 255).astype(uint8).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_ROOT = Path("/data/DiligentPi-Outputs")
DEFAULT_SRC = DEFAULT_ROOT / "1-10"
DEFAULT_DST = DEFAULT_ROOT / "1-10_png"
FAMILIES = ("d", "g")


def npy_to_png(arr: np.ndarray) -> Image.Image:
    """Map a saved normal npy to a uint8 RGB PNG."""
    x = np.asarray(arr)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB, got shape {x.shape}")

    if np.issubdtype(x.dtype, np.floating):
        # Saved preds are [0, 1]; tolerate slight overshoot. Remap [-1, 1] if needed.
        if float(x.min()) < -0.05:
            x = x * 0.5 + 0.5
        x = np.clip(x, 0.0, 1.0)
        rgb = (x * 255.0).astype(np.uint8)
    else:
        rgb = np.clip(x, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def convert_tree(src_root: Path, dst_root: Path, families: tuple[str, ...] = FAMILIES) -> tuple[int, int]:
    converted = 0
    skipped = 0
    for family in families:
        fam_src = src_root / family
        if not fam_src.is_dir():
            print(f"Skip missing family: {fam_src}")
            continue
        for obj_dir in sorted(p for p in fam_src.iterdir() if p.is_dir()):
            out_dir = dst_root / family / obj_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for npy_path in sorted(obj_dir.glob("*.npy")):
                out_path = out_dir / f"{npy_path.stem}.png"
                try:
                    arr = np.load(npy_path)
                    npy_to_png(arr).save(out_path)
                    converted += 1
                except Exception as exc:  # noqa: BLE001 — keep batch going; report failures
                    print(f"FAIL {npy_path}: {exc}")
                    skipped += 1
    return converted, skipped


def parse_args():
    p = argparse.ArgumentParser(description="Convert DiligentPi-Outputs 1-10/*.npy → 1-10_png/*.png")
    p.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"Source 1-10 root (default: {DEFAULT_SRC})",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help=f"Destination 1-10_png root (default: {DEFAULT_DST})",
    )
    p.add_argument(
        "--families",
        nargs="+",
        default=list(FAMILIES),
        help="Subfolders under src to convert (default: d g)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.src.is_dir():
        raise SystemExit(f"Source not found: {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)
    print(f"SRC={args.src}")
    print(f"DST={args.dst}")
    print(f"families={args.families}")
    n_ok, n_fail = convert_tree(args.src, args.dst, tuple(args.families))
    print(f"Done. converted={n_ok} failed={n_fail}")


if __name__ == "__main__":
    main()
