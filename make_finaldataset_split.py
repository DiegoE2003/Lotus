#!/usr/bin/env python3
"""Create a stratified train/val stem split for finaldataset.

Keeps GT vs syn proportions in both splits so you can score val as:
  all / gt_only / syn_only  (held-out only — honest metrics)

Example:
  python make_finaldataset_split.py \\
    --data_dir .../finaldataset \\
    --output_dir .../finaldataset/splits/seed42_val20 \\
    --val_ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def list_stems(data_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir()
        and (p / "rgb.png").is_file()
        and (p / "normal_map.png").is_file()
        and (p / "mask.png").is_file()
    )


def stratified_split(stems: list[str], val_ratio: float, seed: int):
    rng = random.Random(seed)
    gt = [s for s in stems if "_syn_" not in s]
    syn = [s for s in stems if "_syn_" in s]
    rng.shuffle(gt)
    rng.shuffle(syn)

    def split_group(group: list[str]):
        n_val = max(1, int(round(len(group) * val_ratio))) if group else 0
        if len(group) <= 1:
            n_val = 0  # keep tiny groups in train
        n_val = min(n_val, max(0, len(group) - 1))  # keep ≥1 in train if possible
        val = sorted(group[:n_val])
        train = sorted(group[n_val:])
        return train, val

    gt_train, gt_val = split_group(gt)
    syn_train, syn_val = split_group(syn)
    train = sorted(gt_train + syn_train)
    val = sorted(gt_val + syn_val)
    return {
        "train": train,
        "val": val,
        "train_gt": gt_train,
        "train_syn": syn_train,
        "val_gt": gt_val,
        "val_syn": syn_val,
    }


def write_list(path: Path, stems: list[str]):
    path.write_text("\n".join(stems) + ("\n" if stems else ""))


def main():
    p = argparse.ArgumentParser(description="Stratified train/val split for finaldataset")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stems = list_stems(data_dir)
    if not stems:
        raise SystemExit(f"No samples in {data_dir}")

    parts = stratified_split(stems, args.val_ratio, args.seed)
    write_list(out / "train_stems.txt", parts["train"])
    write_list(out / "val_stems.txt", parts["val"])

    info = {
        "data_dir": str(data_dir),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "protocol": (
            "Train on train_stems (GT+syn combined). "
            "Score ONLY val_stems with splits all / gt_only / syn_only."
        ),
        "counts": {
            "all": len(stems),
            "train": len(parts["train"]),
            "val": len(parts["val"]),
            "train_gt": len(parts["train_gt"]),
            "train_syn": len(parts["train_syn"]),
            "val_gt": len(parts["val_gt"]),
            "val_syn": len(parts["val_syn"]),
        },
    }
    (out / "split_info.json").write_text(json.dumps(info, indent=2) + "\n")

    print("=== stratified split ===")
    print(json.dumps(info["counts"], indent=2))
    print(f"Wrote {out / 'train_stems.txt'}")
    print(f"Wrote {out / 'val_stems.txt'}")
    print(f"Wrote {out / 'split_info.json'}")
    print(info["protocol"])


if __name__ == "__main__":
    main()
