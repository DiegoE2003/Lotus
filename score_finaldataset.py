#!/usr/bin/env python3
"""Score predictions against finaldataset (All / GT-only / Syn-only).

Angular error + aggregate metrics use official Lotus helpers:
  evaluation/util/normal_utils.py
    - compute_normal_error
    - compute_normal_metrics
(same path as eval.py → evaluation_normal).

Run inference ONCE on all samples. Split scores afterward — no second run.

  all       — every stem
  gt_only   — stems WITHOUT '_syn_'
  syn_only  — stems WITH '_syn_'

Primary paper-style columns (Lotus evaluation_normal):
  MAE       = mean  (= Lotus 'mean')
  MED       = median (= Lotus 'median')
  <11.25    = a3     (= Lotus 'a3')

Also reports per-image MAE average and angular RMSE.

Expects:
  data_dir/{stem}/rgb.png
  data_dir/{stem}/normal_map.png
  data_dir/{stem}/mask.png

Predictions:
  prediction_dir/{stem}.npy   (preferred)
  prediction_dir/{stem}.png
  prediction_dir/normal/{stem}.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tabulate import tabulate

from evaluation.util.normal_utils import compute_normal_error, compute_normal_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Score finaldataset (All / GT / Syn)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--prediction_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--stems_file",
        default=None,
        help="Optional text file of stems to score (e.g. val_stems.txt). "
        "Within that set still reports all / gt_only / syn_only.",
    )
    return p.parse_args()


def resolve_pred_dir(prediction_dir: Path) -> Path:
    if (prediction_dir / "normal").is_dir():
        has_flat = list(prediction_dir.glob("*.npy")) or list(prediction_dir.glob("*.png"))
        if not has_flat:
            return prediction_dir / "normal"
    return prediction_dir


def load_pred(pred_dir: Path, stem: str) -> np.ndarray:
    """Load prediction as HWC float32 in [-1, 1]."""
    npy = pred_dir / f"{stem}.npy"
    if npy.is_file():
        arr = np.load(npy).astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        if arr.min() >= -0.05 and arr.max() <= 1.05:
            arr = arr * 2.0 - 1.0
        return arr

    png = pred_dir / f"{stem}.png"
    if not png.is_file():
        raise FileNotFoundError(stem)
    arr = cv2.cvtColor(cv2.imread(str(png), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    return arr.astype(np.float32) / 255.0 * 2.0 - 1.0


def hwc_to_bchw(arr: np.ndarray) -> torch.Tensor:
    """(H, W, 3) numpy → (1, 3, H, W) float torch — layout expected by Lotus."""
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).float()


def summarize(img_maes: list[float], pix_errs: list[torch.Tensor]) -> dict:
    """Pixel-pooled metrics via Lotus compute_normal_metrics."""
    if not img_maes:
        nan = float("nan")
        return {
            "num_images": 0,
            "per_image_mae_avg": nan,
            "mae_deg": nan,
            "med_deg": nan,
            "rmse_deg": nan,
            "a_11_25": nan,
            "a_5": nan,
            "a_7_5": nan,
            "a_22_5": nan,
            "a_30": nan,
            "num_pixels": 0,
        }
    total = torch.cat(pix_errs, dim=0)
    m = compute_normal_metrics(total)
    return {
        "num_images": len(img_maes),
        "per_image_mae_avg": float(np.mean(img_maes)),
        "mae_deg": float(m["mean"]),
        "med_deg": float(m["median"]),
        "rmse_deg": float(m["rmse"]),
        "a_11_25": float(m["a3"]),
        "a_5": float(m["a1"]),
        "a_7_5": float(m["a2"]),
        "a_22_5": float(m["a4"]),
        "a_30": float(m["a5"]),
        "num_pixels": int(total.numel()),
    }


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    pred_dir = resolve_pred_dir(Path(args.prediction_dir))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image = {}
    buckets = {
        "all": {"img_mae": [], "pix": []},
        "gt_only": {"img_mae": [], "pix": []},
        "syn_only": {"img_mae": [], "pix": []},
    }
    missing = []

    stems = sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and (p / "normal_map.png").is_file() and (p / "mask.png").is_file()
    )
    if args.stems_file:
        allow = {
            ln.strip()
            for ln in Path(args.stems_file).read_text().splitlines()
            if ln.strip()
        }
        stems = [s for s in stems if s in allow]
        print(f"Scoring {len(stems)} stems from {args.stems_file}")

    for stem in stems:
        sample_dir = data_dir / stem
        try:
            pred = load_pred(pred_dir, stem)
        except FileNotFoundError:
            missing.append(stem)
            continue
        except Exception as e:
            print(f"WARN: bad pred {stem}: {e}")
            missing.append(stem)
            continue

        gt = cv2.cvtColor(
            cv2.imread(str(sample_dir / "normal_map.png"), cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)
        if gt.max() > 1.5:
            gt = gt / 255.0 * 2.0 - 1.0

        mask = cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"WARN: bad mask {stem}")
            continue
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
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Same valid-pixel rule as before; error itself is Lotus compute_normal_error
        valid_np = mask & (np.linalg.norm(gt, axis=-1) > 0.5)
        if not np.any(valid_np):
            print(f"WARN: no valid pixels for {stem}")
            continue

        pred_t = hwc_to_bchw(pred)
        gt_t = hwc_to_bchw(gt)
        # Matches evaluation_normal: compute_normal_error then index by mask
        pred_error = compute_normal_error(pred_t, gt_t)  # (1, 1, H, W)
        valid = torch.from_numpy(valid_np.astype(bool))
        err_v = pred_error[0, 0][valid]  # 1D, degrees
        if err_v.numel() == 0:
            print(f"WARN: no valid pixels for {stem}")
            continue

        img_metrics = compute_normal_metrics(err_v)
        mae = float(img_metrics["mean"])
        is_syn = "_syn_" in stem
        per_image[stem] = {
            "mae_deg": mae,
            "median_deg": float(img_metrics["median"]),
            "rmse_deg": float(img_metrics["rmse"]),
            "a_11_25": float(img_metrics["a3"]),
            "num_pixels": int(err_v.numel()),
            "is_syn": is_syn,
        }
        for b in ("all", "syn_only" if is_syn else "gt_only"):
            buckets[b]["img_mae"].append(mae)
            buckets[b]["pix"].append(err_v.detach().cpu().reshape(-1))

    splits = {name: summarize(b["img_mae"], b["pix"]) for name, b in buckets.items()}
    summary = {
        "data_dir": str(data_dir),
        "prediction_dir": str(pred_dir),
        "stems_file": args.stems_file,
        "missing_predictions": missing,
        "metric_notes": {
            "source": "evaluation.util.normal_utils.compute_normal_error + compute_normal_metrics",
            "MAE": "Lotus 'mean' (pixel-pooled mean angular error, degrees)",
            "MED": "Lotus 'median'",
            "<11.25": "Lotus 'a3' (% pixels with error < 11.25 deg)",
        },
        "per_image": per_image,
        "splits": splits,
    }

    json_path = out_dir / "eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    table = []
    for name in ("all", "gt_only", "syn_only"):
        s = splits[name]
        table.append(
            [
                name,
                s["num_images"],
                f"{s['mae_deg']:.3f}",
                f"{s['med_deg']:.3f}",
                f"{s['a_11_25']:.3f}",
                f"{s['rmse_deg']:.3f}",
                f"{s['per_image_mae_avg']:.3f}",
            ]
        )

    txt = tabulate(
        table,
        headers=[
            "split",
            "N_images",
            "MAE",
            "MED",
            "<11.25",
            "RMSE",
            "per_image_MAE",
        ],
        tablefmt="github",
    )
    txt_path = out_dir / "eval_metrics.txt"
    with open(txt_path, "w") as f:
        f.write(txt + "\n")
        f.write(
            "\nMAE/MED/<11.25>/RMSE from Lotus normal_utils "
            "(compute_normal_error + compute_normal_metrics).\n"
            "per_image_MAE is equal-weight average of per-image MAEs.\n"
            "gt_only = no '_syn_' in stem; syn_only = '_syn_' in stem.\n"
        )
        if args.stems_file:
            f.write(f"\nstems_file filter: {args.stems_file}\n")
        if missing:
            f.write("\nMissing predictions:\n")
            f.write("\n".join(missing) + "\n")

    print("\n=== finaldataset scores ===")
    print(txt)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")
    if missing:
        print(f"Missing predictions: {len(missing)}")


if __name__ == "__main__":
    main()
