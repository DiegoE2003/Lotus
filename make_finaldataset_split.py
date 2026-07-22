#!/usr/bin/env python3
"""Create a stratified train/val/test stem split for finaldataset.

Keeps GT vs syn proportions in each split. Protocol:
  - Train on train_stems (GT+syn combined)
  - Tune / early-stop style checks on val_stems
  - Report final numbers ONLY on test_stems (once)

Example (70/15/15):
  python make_finaldataset_split.py \\
    --data_dir .../finaldataset \\
    --output_dir .../finaldataset/splits/seed42_tvt_70_15_15 \\
    --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --seed 42

Legacy train/val only (no test):
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


def _split_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate n items into train/val/test counts that sum to n."""
    r_train, r_val, r_test = ratios
    if n == 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        # Prefer train + one holdout (val); no test
        return 1, 1, 0

    n_test = int(round(n * r_test))
    n_val = int(round(n * r_val))
    n_train = n - n_val - n_test

    # Ensure train gets at least 1 when possible
    if n_train < 1:
        deficit = 1 - n_train
        n_train = 1
        if n_test >= n_val and n_test >= deficit:
            n_test -= deficit
        else:
            n_val = max(0, n_val - deficit)

    # Prefer non-empty val/test when ratio > 0 and n is large enough
    if r_val > 0 and n_val == 0 and n_train > 1:
        n_val, n_train = 1, n_train - 1
    if r_test > 0 and n_test == 0 and n_train > 1:
        n_test, n_train = 1, n_train - 1

    assert n_train + n_val + n_test == n
    return n_train, n_val, n_test


def stratified_split_tvt(
    stems: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train+val+test ratios must sum to 1, got {total}")

    rng = random.Random(seed)
    gt = [s for s in stems if "_syn_" not in s]
    syn = [s for s in stems if "_syn_" in s]
    rng.shuffle(gt)
    rng.shuffle(syn)

    def split_group(group: list[str]):
        n_train, n_val, n_test = _split_counts(
            len(group), (train_ratio, val_ratio, test_ratio)
        )
        # order after shuffle: test, val, train (holdouts first for stability)
        test = sorted(group[:n_test])
        val = sorted(group[n_test : n_test + n_val])
        train = sorted(group[n_test + n_val :])
        assert len(train) == n_train
        return train, val, test

    gt_train, gt_val, gt_test = split_group(gt)
    syn_train, syn_val, syn_test = split_group(syn)

    return {
        "train": sorted(gt_train + syn_train),
        "val": sorted(gt_val + syn_val),
        "test": sorted(gt_test + syn_test),
        "train_gt": gt_train,
        "train_syn": syn_train,
        "val_gt": gt_val,
        "val_syn": syn_val,
        "test_gt": gt_test,
        "test_syn": syn_test,
    }


def stratified_split(stems: list[str], val_ratio: float, seed: int):
    """Legacy train/val-only (test empty). Kept for older call sites."""
    parts = stratified_split_tvt(stems, 1.0 - val_ratio, val_ratio, 0.0, seed)
    return {k: v for k, v in parts.items() if not k.startswith("test")}


def write_list(path: Path, stems: list[str]):
    path.write_text("\n".join(stems) + ("\n" if stems else ""))


def main():
    p = argparse.ArgumentParser(description="Stratified train/val/test split for finaldataset")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    # New TVT API
    p.add_argument("--train_ratio", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--test_ratio", type=float, default=None)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stems = list_stems(data_dir)
    if not stems:
        raise SystemExit(f"No samples in {data_dir}")

    # Defaults: if only --val_ratio given (legacy), no test. Else 70/15/15.
    if args.train_ratio is None and args.test_ratio is None:
        val_ratio = 0.2 if args.val_ratio is None else args.val_ratio
        train_ratio, test_ratio = 1.0 - val_ratio, 0.0
        val_ratio = val_ratio
    else:
        train_ratio = 0.7 if args.train_ratio is None else args.train_ratio
        val_ratio = 0.15 if args.val_ratio is None else args.val_ratio
        test_ratio = 0.15 if args.test_ratio is None else args.test_ratio

    parts = stratified_split_tvt(stems, train_ratio, val_ratio, test_ratio, args.seed)
    write_list(out / "train_stems.txt", parts["train"])
    write_list(out / "val_stems.txt", parts["val"])
    if parts["test"]:
        write_list(out / "test_stems.txt", parts["test"])

    info = {
        "data_dir": str(data_dir),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": args.seed,
        "protocol": (
            "Train on train_stems (GT+syn). "
            "Use val_stems for model selection / ablations. "
            "Score test_stems only for final reported metrics (all / gt_only / syn_only)."
            if parts["test"]
            else (
                "Train on train_stems (GT+syn combined). "
                "Score ONLY val_stems with splits all / gt_only / syn_only."
            )
        ),
        "counts": {
            "all": len(stems),
            "train": len(parts["train"]),
            "val": len(parts["val"]),
            "test": len(parts["test"]),
            "train_gt": len(parts["train_gt"]),
            "train_syn": len(parts["train_syn"]),
            "val_gt": len(parts["val_gt"]),
            "val_syn": len(parts["val_syn"]),
            "test_gt": len(parts["test_gt"]),
            "test_syn": len(parts["test_syn"]),
        },
    }
    (out / "split_info.json").write_text(json.dumps(info, indent=2) + "\n")

    print("=== stratified split ===")
    print(json.dumps(info["counts"], indent=2))
    print(f"Wrote {out / 'train_stems.txt'}")
    print(f"Wrote {out / 'val_stems.txt'}")
    if parts["test"]:
        print(f"Wrote {out / 'test_stems.txt'}")
    print(f"Wrote {out / 'split_info.json'}")
    print(info["protocol"])


if __name__ == "__main__":
    main()
