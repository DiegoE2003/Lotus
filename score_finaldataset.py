#!/usr/bin/env python3
"""Score disc predictions for paper tables.

Scoring math is ONLY the shared helper that copies evaluation_normal:
  evaluation/util/evaluation_normal_score.py
    → compute_normal_error → pred_error[mask] → compute_normal_metrics

After FT use --split_dir .../seed42_tvt_70_15_15 (test_stems only).
Also reports all / gt_only / syn_only buckets (lab split, same math each).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tabulate import tabulate
from torchvision.transforms.functional import resize

from evaluation.util.evaluation_normal_score import (
    accumulate_normal_errors,
    metrics_from_errors,
)


def parse_args():
    p = argparse.ArgumentParser(description="Score finaldataset (Lotus evaluation_normal math)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--prediction_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--split_dir",
        default=None,
        help="If set (and no --stems_file), score test_stems.txt only.",
    )
    p.add_argument("--stems_file", default=None, help="Explicit stem list (overrides --split_dir).")
    return p.parse_args()


def resolve_pred_dir(prediction_dir: Path) -> Path:
    if (prediction_dir / "normal").is_dir():
        has_flat = list(prediction_dir.glob("*.npy")) or list(prediction_dir.glob("*.png"))
        if not has_flat:
            return prediction_dir / "normal"
    return prediction_dir


def load_pred(pred_dir: Path, stem: str) -> torch.Tensor:
    """Load pred as (1,3,H,W) in [-1,1]."""
    npy = pred_dir / f"{stem}.npy"
    if npy.is_file():
        arr = np.load(npy).astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        if arr.min() >= -0.05 and arr.max() <= 1.05:
            arr = arr * 2.0 - 1.0
    else:
        png = pred_dir / f"{stem}.png"
        if not png.is_file():
            raise FileNotFoundError(stem)
        arr = cv2.cvtColor(cv2.imread(str(png), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).float()


def load_gt_mask(sample_dir: Path) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """(1,3,H,W) GT + (1,1,H,W) mask for accumulate_normal_errors."""
    gt = cv2.cvtColor(
        cv2.imread(str(sample_dir / "cleaned_normalmap.png"), cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32)
    if gt.max() > 1.5:
        gt = gt / 255.0 * 2.0 - 1.0
    gt_t = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).unsqueeze(0).float()

    mask = cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None
    mask = mask > 0
    if mask.shape[:2] != gt.shape[:2]:
        mask = (
            cv2.resize(
                mask.astype(np.uint8),
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
    gt_norm_mask = torch.from_numpy(mask.astype(bool))[None, None]
    return gt_t, gt_norm_mask


def resolve_stems(data_dir: Path, stems_file: str | None, split_dir: str | None):
    stems = sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and (p / "cleaned_normalmap.png").is_file() and (p / "mask.png").is_file()
    )
    path = stems_file
    if path is None and split_dir:
        test_path = Path(split_dir) / "test_stems.txt"
        if not test_path.is_file():
            raise SystemExit(f"Missing {test_path}")
        path = str(test_path)
        print(f"Using test split from {path}")
    if path:
        allow = {ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()}
        stems = [s for s in stems if s in allow]
        print(f"Scoring {len(stems)} stems from {path}")
    return stems, path


def score_stem_list(data_dir: Path, pred_dir: Path, stems: list[str]) -> tuple[dict, dict, list]:
    """Pool errors with stock accumulate_normal_errors; return splits + per_image + missing."""
    pools = {"all": None, "gt_only": None, "syn_only": None}
    counts = {"all": 0, "gt_only": 0, "syn_only": 0}
    per_image = {}
    missing = []

    for stem in stems:
        try:
            pred = load_pred(pred_dir, stem)
        except FileNotFoundError:
            missing.append(stem)
            continue

        gt, gt_mask = load_gt_mask(data_dir / stem)
        if gt is None:
            continue

        if pred.shape[-2:] != gt.shape[-2:]:
            pred = resize(pred, gt.shape[-2:], antialias=True)

        # one-image pool via same helper as evaluation_normal
        img_errs = accumulate_normal_errors(pred, gt, gt_mask, None)
        if img_errs is None or img_errs.numel() == 0:
            continue

        img_m = metrics_from_errors(img_errs)
        is_syn = "_syn_" in stem
        per_image[stem] = {
            "mean": float(img_m["mean"]),
            "median": float(img_m["median"]),
            "a3": float(img_m["a3"]),
            "is_syn": is_syn,
        }

        for key in ("all", "syn_only" if is_syn else "gt_only"):
            pools[key] = accumulate_normal_errors(pred, gt, gt_mask, pools[key])
            counts[key] += 1

    splits = {}
    for name, pool in pools.items():
        m = metrics_from_errors(pool)
        if m is None:
            splits[name] = {
                "num_images": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "rmse": float("nan"),
                "a1": float("nan"),
                "a2": float("nan"),
                "a3": float("nan"),
                "a4": float("nan"),
                "a5": float("nan"),
            }
        else:
            splits[name] = {"num_images": counts[name], **{k: float(v) for k, v in m.items()}}
    return splits, per_image, missing


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    pred_dir = resolve_pred_dir(Path(args.prediction_dir))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems, stems_file = resolve_stems(data_dir, args.stems_file, args.split_dir)
    splits, per_image, missing = score_stem_list(data_dir, pred_dir, stems)

    summary = {
        "data_dir": str(data_dir),
        "prediction_dir": str(pred_dir),
        "stems_file": stems_file,
        "split_dir": args.split_dir,
        "score_source": "evaluation.util.evaluation_normal_score (from evaluation_normal)",
        "missing_predictions": missing,
        "per_image": per_image,
        "splits": splits,
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    table = [
        [name, splits[name]["num_images"], f"{splits[name]['mean']:.3f}",
         f"{splits[name]['median']:.3f}", f"{splits[name]['a3']:.3f}", f"{splits[name]['rmse']:.3f}"]
        for name in ("all", "gt_only", "syn_only")
    ]
    txt = tabulate(
        table,
        headers=["split", "N", "mean", "median", "a3(<11.25)", "rmse"],
        tablefmt="github",
    )
    (out_dir / "eval_metrics.txt").write_text(
        txt
        + "\n\nmean/median/a3/rmse = Lotus compute_normal_metrics "
        "(via evaluation_normal_score).\n"
        + (f"stems_file: {stems_file}\n" if stems_file else "")
        + (("missing:\n" + "\n".join(missing) + "\n") if missing else "")
    )

    print("\n=== finaldataset scores ===")
    print(txt)
    print(f"\nSaved: {out_dir / 'eval_metrics.json'}")
    if missing:
        print(f"Missing: {len(missing)}")


if __name__ == "__main__":
    main()
