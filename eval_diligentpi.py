#!/usr/bin/env python3
"""Evaluate Lotus normals on DiLiGenT-Pi and report MAE (mean angular error, degrees).

DiLiGenT-Pi paper metric: MAngE = mean angular error between predicted and GT normals
on valid (masked) pixels. Lotus reports the same quantity as ``mean`` in evaluation_normal.

Expected layout (official release on maul):
  /data/DiLiGent-Pi/DiLiGenT-Pi_release/
    DiLiGenT-Pi_release_png/<Object>/   # 001.png ... 100.png, mask.png
    DiLiGenT-Pi_release_mat/<Object>/   # optional float .mat images

GT normals (Normal_gt.png / Normal_gt.mat) are NOT in the public PNG/MAT
image release — download the evaluation GT from the DiLiGenT-Pi website and
pass ``--gt_dir`` to that folder (same per-object subfolder layout).

Example (all metallic coins):
  conda activate lotus
  cd reproducing/Lotus/Lotus
  python eval_diligentpi.py \\
    --data_dir /data/DiLiGent-Pi/DiLiGenT-Pi_release \\
    --gt_dir /path/to/DiLiGenT-Pi_GT \\
    --output_dir output/diligentpi_lotus_g_metallic \\
    --objects FLOWER BIRD RHINO LIONS QUEEN CRAB SHIP PARA SAIL FISH \\
    --pretrained_model_name_or_path jingheya/lotus-normal-g-v1-1 \\
    --mode generation --half_precision --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tabulate import tabulate
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm

from evaluation.util import normal_utils
from pipeline import LotusDPipeline, LotusGPipeline
from utils.seed_all import seed_all

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# 30 objects from DiLiGenT-Pi (ICCV 2023), Table 2
DILIGENTPI_OBJECTS = [
    "FLOWER", "BIRD", "RHINO", "LIONS", "QUEEN", "CRAB", "SHIP", "PARA", "SAIL", "FISH",
    "TREE", "OCEAN", "LUNG", "BEAR", "TV", "SUN", "TAICHI", "WAVE", "ASTRO", "WHALE",
    "BAGUA-T", "LOTUS-T", "LION-T", "PANDA-T", "CLOUD-T",
    "BAGUA-R", "LOTUS-R", "LION-R", "PANDA-R", "CLOUD-R",
]

MATERIAL_GROUPS = {
    "metallic": DILIGENTPI_OBJECTS[0:10],
    "specular": DILIGENTPI_OBJECTS[10:20],
    "translucent": DILIGENTPI_OBJECTS[20:25],
    "rough": DILIGENTPI_OBJECTS[25:30],
}

CANONICAL_BY_KEY = {
    name.lower().replace("_", "-"): name for name in DILIGENTPI_OBJECTS
}

IMAGE_SUBDIR_CANDIDATES = (
    "DiLiGenT-Pi_release_png",
    "DiLiGenT-Pi_release_mat",
    "pmsData",
)

GT_CANDIDATES = ("Normal_gt.png", "normal_gt.png", "Normal_gt.mat", "normal_gt.mat")
MASK_CANDIDATES = ("mask.png", "Mask.png")
SKIP_IMAGE_NAMES = {
    "mask.png", "mask.PNG", "Normal_gt.png", "normal_gt.png",
    "Normal_gt.mat", "normal_gt.mat",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Lotus MAE evaluation on DiLiGenT-Pi")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="DiLiGenT-Pi release root, e.g. /data/DiLiGent-Pi/DiLiGenT-Pi_release",
    )
    parser.add_argument(
        "--image_subdir",
        type=str,
        default=None,
        help="Subfolder with lit images (default: auto DiLiGenT-Pi_release_png).",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Folder with per-object Normal_gt files (if not inside image folders).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save predictions and eval_metrics.json / .txt",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="jingheya/lotus-normal-g-v1-1",
        help="Lotus checkpoint (HF id or local path). Ignored if --prediction_dir is set.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generation",
        choices=("generation", "regression"),
        help="Lotus G (generation) or Lotus D (regression).",
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default=None,
        help="Score existing predictions only. PNGs named <object>_norm.png",
    )
    parser.add_argument(
        "--light_index",
        type=int,
        default=0,
        help="Which lit image to use per object (0-based index into filenames.txt or sorted PNG list).",
    )
    parser.add_argument(
        "--objects",
        type=str,
        nargs="*",
        default=None,
        help="Subset of object names. Default: all discovered objects.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Max edge resolution for Lotus inference (None = pipeline default).",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save colored normal predictions under output_dir/normal_vis/",
    )
    return parser.parse_args()


def canonical_object_name(name: str) -> str:
    key = name.lower().replace("_", "-")
    return CANONICAL_BY_KEY.get(key, name.upper())


def resolve_image_root(data_dir: str, image_subdir: str | None) -> Path:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir not found: {root}")

    if image_subdir:
        image_root = root / image_subdir
        if not image_root.is_dir():
            raise FileNotFoundError(f"image_subdir not found: {image_root}")
        return image_root

    for subdir in IMAGE_SUBDIR_CANDIDATES:
        candidate = root / subdir
        if candidate.is_dir():
            return candidate

    # data_dir already points at object folders (Flower/, Queen/, ...)
    if any(child.is_dir() and _find_file(child, MASK_CANDIDATES) for child in root.iterdir()):
        return root

    raise FileNotFoundError(
        f"Could not find image folders under {root}. "
        f"Expected one of {IMAGE_SUBDIR_CANDIDATES} or per-object subdirs with mask.png."
    )


def resolve_gt_root(data_dir: str, gt_dir: str | None, image_root: Path) -> Path:
    if gt_dir is None:
        return image_root
    root = Path(gt_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"gt_dir not found: {root}")
    return root


def _object_key(name: str) -> str:
    return name.lower().replace("_", "-")


def find_object_dir(data_root: Path, obj_name: str) -> Path:
    canonical = canonical_object_name(obj_name)
    title = "-".join(part.capitalize() for part in _object_key(canonical).split("-"))
    for candidate in (
        data_root / obj_name,
        data_root / canonical,
        data_root / canonical.upper(),
        data_root / canonical.lower(),
        data_root / title,
    ):
        if candidate.is_dir():
            return candidate

    target = _object_key(canonical)
    for child in data_root.iterdir():
        if child.is_dir() and _object_key(child.name) == target:
            return child

    raise FileNotFoundError(f"Object folder not found for {obj_name} under {data_root}")


def discover_objects(image_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return [canonical_object_name(name) for name in requested]

    found = []
    for name in DILIGENTPI_OBJECTS:
        try:
            find_object_dir(image_root, name)
            found.append(name)
        except FileNotFoundError:
            continue
    if found:
        return found

    for child in sorted(image_root.iterdir()):
        if not child.is_dir():
            continue
        if _find_file(child, MASK_CANDIDATES):
            found.append(canonical_object_name(child.name))
    if not found:
        raise FileNotFoundError(
            f"No DiLiGenT-Pi object folders found under {image_root}."
        )
    return found


def _find_file(obj_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = obj_dir / name
        if path.is_file():
            return path
    lower_map = {p.name.lower(): p for p in obj_dir.iterdir() if p.is_file()}
    for name in names:
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def load_mask(obj_dir: Path) -> np.ndarray:
    mask_path = _find_file(obj_dir, MASK_CANDIDATES)
    if mask_path is None:
        raise FileNotFoundError(f"mask not found in {obj_dir}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    return mask > 0


def load_gt_normal(gt_dir: Path) -> np.ndarray:
    gt_path = _find_file(gt_dir, GT_CANDIDATES)
    if gt_path is None:
        raise FileNotFoundError(
            f"GT normal not found in {gt_dir}. "
            "The PNG/MAT image release does not include Normal_gt — download the "
            "evaluation GT from https://photometricstereo.github.io/diligentpi.html "
            "and pass --gt_dir."
        )

    if gt_path.suffix.lower() == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError(
                "Normal_gt.mat requires scipy (`pip install scipy`) or use Normal_gt.png."
            ) from exc
        mat = loadmat(str(gt_path))
        for key in ("Normal_gt", "normal_gt", "n_gt", "normal"):
            if key in mat:
                normal = np.asarray(mat[key], dtype=np.float32)
                break
        else:
            raise KeyError(f"No normal array in {gt_path}; keys: {list(mat.keys())}")
        if normal.ndim == 3 and normal.shape[0] == 3:
            normal = np.transpose(normal, (1, 2, 0))
    else:
        normal = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if normal is None:
            raise RuntimeError(f"Failed to read GT normal: {gt_path}")
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        if normal.dtype == np.uint16:
            normal = normal.astype(np.float32) / 65535.0 * 2.0 - 1.0
        else:
            normal = normal.astype(np.float32) / 255.0 * 2.0 - 1.0

    return normal.astype(np.float32)


def list_lit_images(obj_dir: Path) -> list[Path]:
    names_file = obj_dir / "filenames.txt"
    if names_file.is_file():
        with open(names_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        paths = [obj_dir / n for n in names]
        paths = [p for p in paths if p.is_file()]
        if paths:
            return paths

    paths = []
    for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg"):
        paths.extend(obj_dir.glob(ext))
    paths = sorted(
        p for p in paths
        if p.name not in SKIP_IMAGE_NAMES and "err" not in p.name.lower()
    )
    return paths


def pick_input_image(obj_dir: Path, light_index: int) -> Path:
    images = list_lit_images(obj_dir)
    if not images:
        raise FileNotFoundError(f"No lit images found in {obj_dir}")
    if light_index < 0 or light_index >= len(images):
        raise IndexError(
            f"light_index={light_index} out of range for {obj_dir.name} "
            f"({len(images)} images)"
        )
    return images[light_index]


def load_rgb_for_lotus(image_path: Path) -> torch.Tensor:
    """Return float tensor (1, 3, H, W) in [-1, 1]."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 65535.0 * 255.0).clip(0, 255)
    else:
        img = img.astype(np.float32)

    if img.max() > 255.0:
        img = img / img.max() * 255.0

    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor / 127.5 - 1.0
    return tensor


def load_prediction_png(pred_path: Path, device: torch.device) -> torch.Tensor:
    norm = cv2.cvtColor(cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    norm = (norm.astype(np.float32) / 255.0) * 2.0 - 1.0
    return torch.tensor(norm).permute(2, 0, 1).unsqueeze(0).to(device)


def save_prediction_png(pred_hw3: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = ((pred_hw3 * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(vis).save(path)


def build_pipeline(args, device: torch.device):
    dtype = torch.float16 if args.half_precision else torch.float32
    if args.mode == "generation":
        pipe = LotusGPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype)
    else:
        pipe = LotusDPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def predict_normal(pipe, rgb: torch.Tensor, args, device: torch.device) -> torch.Tensor:
    rgb = rgb.to(device)
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device.type)

    with autocast_ctx:
        task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

        out = pipe(
            rgb_in=rgb,
            prompt="",
            num_inference_steps=1,
            output_type="pt",
            timesteps=[args.timestep],
            task_emb=task_emb,
            processing_res=args.processing_res if args.processing_res else 0,
        ).images[0]
        pred = (out * 2.0 - 1.0).unsqueeze(0)
    return pred


def compute_object_errors(
    pred_norm: torch.Tensor,
    gt_norm: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    gt = torch.tensor(gt_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    mask_t = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(device)

    if pred_norm.shape[-2:] != gt.shape[-2:]:
        pred_norm = resize(pred_norm, gt.shape[-2:], antialias=True)

    gt_norm_mask = torch.linalg.norm(gt, dim=1, keepdim=True) > 0.5
    valid = mask_t & gt_norm_mask

    pred_error = normal_utils.compute_normal_error(pred_norm, gt)
    errors = pred_error[valid]
    metrics = normal_utils.compute_normal_metrics(errors)
    return errors, metrics


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    seed_all(args.seed)

    image_root = resolve_image_root(args.data_dir, args.image_subdir)
    gt_root = resolve_gt_root(args.data_dir, args.gt_dir, image_root)
    objects = discover_objects(image_root, args.objects)
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        (output_dir / "normal_vis").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    logging.info("Image root: %s", image_root)
    logging.info("GT root: %s", gt_root)
    logging.info("Objects (%d): %s", len(objects), ", ".join(objects))

    pipe = None
    if args.prediction_dir is None:
        pipe = build_pipeline(args, device)
        logging.info("Loaded pipeline: %s", args.pretrained_model_name_or_path)
    else:
        logging.info("Scoring predictions from: %s", args.prediction_dir)

    per_object = {}
    all_errors = []

    for obj_name in tqdm(objects, desc="DiLiGenT-Pi"):
        image_dir = find_object_dir(image_root, obj_name)
        gt_dir = find_object_dir(gt_root, obj_name)
        gt_normal = load_gt_normal(gt_dir)
        mask = load_mask(image_dir)
        if mask.shape[:2] != gt_normal.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (gt_normal.shape[1], gt_normal.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

        pred_path = (
            Path(args.prediction_dir) / f"{obj_name}_norm.png"
            if args.prediction_dir
            else pred_dir / f"{obj_name}_norm.png"
        )

        if args.prediction_dir is None:
            img_path = pick_input_image(image_dir, args.light_index)
            rgb = load_rgb_for_lotus(img_path)
            pred_norm = predict_normal(pipe, rgb, args, device)
            pred_hw3 = pred_norm.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            save_prediction_png(pred_hw3, pred_path)
            if args.save_vis:
                save_prediction_png(pred_hw3, output_dir / "normal_vis" / f"{obj_name}.png")
            torch.cuda.empty_cache()
        else:
            if not pred_path.is_file():
                raise FileNotFoundError(f"Missing prediction: {pred_path}")
            pred_norm = load_prediction_png(pred_path, device)

        errors, metrics = compute_object_errors(pred_norm, gt_normal, mask, device)
        per_object[obj_name] = {
            "mae_deg": float(metrics["mean"]),
            "median_deg": float(metrics["median"]),
            "rmse_deg": float(metrics["rmse"]),
            "input_image": str(pick_input_image(image_dir, args.light_index)) if args.prediction_dir is None else None,
        }
        all_errors.append(errors.detach().cpu())

    all_errors_t = torch.cat(all_errors, dim=0)
    global_metrics = normal_utils.compute_normal_metrics(all_errors_t)

    group_mae = {}
    for group_name, group_objs in MATERIAL_GROUPS.items():
        vals = [per_object[o]["mae_deg"] for o in group_objs if o in per_object]
        if vals:
            group_mae[group_name] = float(np.mean(vals))

    summary = {
        "dataset": "DiLiGenT-Pi",
        "metric": "MAE (mean angular error, degrees) = MAngE",
        "num_objects": len(per_object),
        "light_index": args.light_index,
        "image_root": str(image_root),
        "gt_root": str(gt_root),
        "model": args.pretrained_model_name_or_path if args.prediction_dir is None else f"predictions:{args.prediction_dir}",
        "mode": args.mode,
        "global_mae_deg": float(global_metrics["mean"]),
        "global_median_deg": float(global_metrics["median"]),
        "global_rmse_deg": float(global_metrics["rmse"]),
        "group_mean_mae_deg": group_mae,
        "per_object_mae_deg": {k: v["mae_deg"] for k, v in per_object.items()},
        "per_object": per_object,
    }

    json_path = output_dir / "eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    table_rows = [[obj, f"{per_object[obj]['mae_deg']:.3f}"] for obj in objects if obj in per_object]
    table_rows.append(["GLOBAL (all pixels)", f"{summary['global_mae_deg']:.3f}"])
    for g, v in group_mae.items():
        table_rows.append([f"{g} avg", f"{v:.3f}"])

    txt = tabulate(table_rows, headers=["Object", "MAE (deg)"], tablefmt="github")
    txt_path = output_dir / "eval_metrics.txt"
    with open(txt_path, "w") as f:
        f.write(txt + "\n")

    print("\n=== DiLiGenT-Pi MAE (Lotus) ===")
    print(txt)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
