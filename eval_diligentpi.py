#!/usr/bin/env python3
"""DiLiGenT-Π runner: Lotus infer + Lotus normal scoring on DiLiGenT folders.

Why this file exists (cannot use stock Lotus alone)
--------------------------------------------------
* ``infer.py`` — flat RGB folder only; no GT / mask / MAngE.
* ``eval.py`` → ``evaluation_normal`` — NYU / ScanNet / iBims / Sintel loaders only.
* ``score_finaldataset.py`` — disc ``finaldataset/{stem}/`` layout only.

This script is **dataset I/O glue** for DiLiGenT-Π. It does not invent a metric:

* Forward path = ``eval.py`` ``gen_normal`` (same pipe call / range / processing_res=0).
* RGB prep = **exact** ``infer.py`` Pillow path (``Image.open→/127.5-1``).
* Angular error = ``evaluation.util.normal_utils`` (same as ``score_finaldataset.py`` /
  ``evaluation_normal``): mask ∩ ‖GT‖ > 0.5, then ``compute_normal_error`` /
  ``compute_normal_metrics``.

Layout
------
  --data_dir  …/DiLiGenT-Pi_release/          # contains DiLiGenT-Pi_release_png/
  --gt_dir    …/DiLiGenT-Pi_gt/               # per-object Normal_gt.mat

Default protocol (monocular over all lights)
-------------------------------------------
For each object (e.g. Astro):
  1. Run Lotus on every lit PNG (~100 light angles).
  2. Compare each prediction to the **same** Normal_gt.
  3. Object MAE = mean of those ~100 per-light MAEs.
Then Avg across objects (paper-style). Use ``--light_index N`` to smoke-test one light only.

Example
-------
  conda activate lotus
  cd reproducing/Lotus/Lotus
  python eval_diligentpi.py \\
    --data_dir /data/DiLiGent-Pi/DiLiGenT-Pi_release \\
    --gt_dir /data/DiLiGent-Pi/DiLiGenT-Pi_release/DiLiGenT-Pi_gt \\
    --output_dir output/diligentpi_lotus_d \\
    --pretrained_model_name_or_path jingheya/lotus-normal-d-v1-1 \\
    --mode regression --half_precision --seed 42
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
from tqdm.auto import tqdm

# Official Lotus metric math (same imports as score_finaldataset.py / evaluation_normal)
from evaluation.util.normal_utils import compute_normal_error, compute_normal_metrics
from pipeline import LotusDPipeline, LotusGPipeline
from utils.seed_all import seed_all

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# ---------------------------------------------------------------------------
# DiLiGenT-Π object list (paper Table 2) — dataset catalog only, not scoring
# ---------------------------------------------------------------------------
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

CANONICAL_BY_KEY = {name.lower().replace("_", "-"): name for name in DILIGENTPI_OBJECTS}

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
    parser = argparse.ArgumentParser(
        description="DiLiGenT-Π: Lotus gen_normal + normal_utils MAngE"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--image_subdir", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="jingheya/lotus-normal-g-v1-1",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generation",
        choices=("generation", "regression"),
        help="Lotus G (generation) or Lotus D (regression) — same as infer.py / eval.py",
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default=None,
        help="Score existing preds only. Expect normal/<OBJ>/<light_stem>.npy "
        "(or flat <OBJ>_<light_stem>.npy).",
    )
    parser.add_argument(
        "--light_index",
        type=int,
        default=None,
        help="If set, only this 0-based light (smoke test). "
        "Default: all lit images per object (~100).",
    )
    parser.add_argument("--objects", type=str, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Passed to pipeline. Default 0 = native res (matches eval.py gen_normal).",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Also write uint8 normal_vis/*.png (infer.py style).",
    )
    return parser.parse_args()


# ========================= DiLiGenT folder helpers =========================

def canonical_object_name(name: str) -> str:
    key = name.lower().replace("_", "-")
    return CANONICAL_BY_KEY.get(key, name.upper())


def _object_key(name: str) -> str:
    return name.lower().replace("_", "-")


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
    if any(child.is_dir() and _find_file(child, MASK_CANDIDATES) for child in root.iterdir()):
        return root
    raise FileNotFoundError(
        f"Could not find image folders under {root}. "
        f"Expected one of {IMAGE_SUBDIR_CANDIDATES}."
    )


def resolve_gt_root(gt_dir: str | None, image_root: Path) -> Path:
    if gt_dir is None:
        return image_root
    root = Path(gt_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"gt_dir not found: {root}")
    return root


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
        if child.is_dir() and _find_file(child, MASK_CANDIDATES):
            found.append(canonical_object_name(child.name))
    if not found:
        raise FileNotFoundError(f"No DiLiGenT-Pi object folders under {image_root}.")
    return found


def list_lit_images(obj_dir: Path) -> list[Path]:
    names_file = obj_dir / "filenames.txt"
    if names_file.is_file():
        with open(names_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        paths = [p for p in (obj_dir / n for n in names) if p.is_file()]
        if paths:
            return paths
    paths = []
    for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg"):
        paths.extend(obj_dir.glob(ext))
    return sorted(
        p for p in paths
        if p.name not in SKIP_IMAGE_NAMES and "err" not in p.name.lower()
    )


def pick_input_image(obj_dir: Path, light_index: int) -> Path:
    images = list_lit_images(obj_dir)
    if not images:
        raise FileNotFoundError(f"No lit images in {obj_dir}")
    if light_index < 0 or light_index >= len(images):
        raise IndexError(
            f"light_index={light_index} out of range for {obj_dir.name} ({len(images)} images)"
        )
    return images[light_index]


def load_mask(obj_dir: Path) -> np.ndarray:
    mask_path = _find_file(obj_dir, MASK_CANDIDATES)
    if mask_path is None:
        raise FileNotFoundError(f"mask not found in {obj_dir}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    return mask > 0


def load_gt_normal(gt_dir: Path) -> np.ndarray:
    """Load DiLiGenT Normal_gt → HWC float32 in roughly [-1, 1]."""
    gt_path = _find_file(gt_dir, GT_CANDIDATES)
    if gt_path is None:
        raise FileNotFoundError(
            f"GT normal not found in {gt_dir}. Pass --gt_dir to DiLiGenT-Pi_gt."
        )
    if gt_path.suffix.lower() == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError(
                "Normal_gt.mat requires scipy (`pip install scipy`)."
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
        # Same decode as score_finaldataset GT / evaluation load_prediction png
        normal = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if normal is None:
            raise RuntimeError(f"Failed to read GT normal: {gt_path}")
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        if normal.dtype == np.uint16:
            normal = normal.astype(np.float32) / 65535.0 * 2.0 - 1.0
        else:
            normal = normal.astype(np.float32) / 255.0 * 2.0 - 1.0
    return normal.astype(np.float32)


# ========================= Lotus infer path (copied) =========================

def load_rgb_like_infer(image_path: Path, device: torch.device) -> torch.Tensor:
    """Exact RGB prep from ``infer.py`` (lines ~183–187).

    Note: DiLiGenT PNGs are 16-bit on disk; Pillow (as used by Lotus ``infer.py``)
    loads them as uint8 RGB. We deliberately follow that same Pillow path so
    inputs match Lotus inference, not an OpenCV uint16 decode.
    """
    # infer.py:
    #   test_image = Image.open(...).convert('RGB')
    #   test_image = np.array(test_image).astype(np.float32)
    #   test_image = torch.tensor(test_image).permute(2,0,1).unsqueeze(0)
    #   test_image = test_image / 127.5 - 1.0
    #   test_image = test_image.to(device)
    test_image = Image.open(image_path).convert("RGB")
    test_image = np.array(test_image).astype(np.float32)
    test_image = torch.tensor(test_image).permute(2, 0, 1).unsqueeze(0)
    test_image = test_image / 127.5 - 1.0
    test_image = test_image.to(device)
    return test_image


@torch.no_grad()
def gen_normal(
    img: torch.Tensor,
    pipe,
    timestep: int,
    processing_res: int = 0,
    prompt: str = "",
    num_inference_steps: int = 1,
) -> torch.Tensor:
    """Byte-for-byte logic from ``eval.py`` nested ``gen_normal`` (~L195–215).

    Intentionally follows **eval.py** (not ``infer.py``):
      - output_type='pt'  (infer uses 'np')
      - processing_res=0 by default  (infer default None → pipeline 768)
      - no ``generator`` kwarg  (infer passes one)

    img: (1, 3, H, W) in [-1, 1]
    returns: (1, 3, H, W) in [-1, 1]
    """
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(pipe.device.type)

    with autocast_ctx:
        task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(pipe.device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

        pred_normal = pipe(
            rgb_in=img,  # [-1,1]
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            output_type="pt",
            timesteps=[timestep],
            task_emb=task_emb,
            processing_res=processing_res,  # eval.py hardcodes 0
        ).images[0]  # [0,1], (3,h,w)
        pred_normal = (pred_normal * 2 - 1.0).unsqueeze(0)  # [-1,1], (1,3,h,w)
    return pred_normal


def build_pipeline(args, device: torch.device):
    """Same pipeline selection as ``infer.py`` / ``eval.py``."""
    dtype = torch.float16 if args.half_precision else torch.float32
    if args.mode == "generation":
        pipe = LotusGPipeline.from_pretrained(
            args.pretrained_model_name_or_path, torch_dtype=dtype
        )
    else:
        pipe = LotusDPipeline.from_pretrained(
            args.pretrained_model_name_or_path, torch_dtype=dtype
        )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ========================= Lotus score path (copied) =========================

def hwc_to_bchw(arr: np.ndarray) -> torch.Tensor:
    """Copied from ``score_finaldataset.py``."""
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).float()


def load_pred_like_score_finaldataset(pred_path: Path) -> np.ndarray:
    """Load prediction as HWC float32 in [-1, 1] (``score_finaldataset.load_pred``)."""
    if pred_path.suffix.lower() == ".npy":
        arr = np.load(pred_path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        # infer.py saves normals in [0, 1]; score_finaldataset remaps that range
        if arr.min() >= -0.05 and arr.max() <= 1.05:
            arr = arr * 2.0 - 1.0
        return arr
    # PNG path used by evaluation_normal load_prediction
    arr = cv2.cvtColor(cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    return arr.astype(np.float32) / 255.0 * 2.0 - 1.0


def score_normals(
    pred_hwc: np.ndarray,
    gt_hwc: np.ndarray,
    mask: np.ndarray,
) -> tuple[torch.Tensor, dict]:
    """Score one object using the same rules as ``score_finaldataset.py``.

    valid = mask & (‖gt‖ > 0.5)
    error = compute_normal_error → compute_normal_metrics
    """
    if pred_hwc.shape[:2] != gt_hwc.shape[:2]:
        pred_hwc = cv2.resize(
            pred_hwc, (gt_hwc.shape[1], gt_hwc.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    if mask.shape[:2] != gt_hwc.shape[:2]:
        mask = (
            cv2.resize(
                mask.astype(np.uint8),
                (gt_hwc.shape[1], gt_hwc.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )

    valid_np = mask & (np.linalg.norm(gt_hwc, axis=-1) > 0.5)
    pred_t = hwc_to_bchw(pred_hwc)
    gt_t = hwc_to_bchw(gt_hwc)
    pred_error = compute_normal_error(pred_t, gt_t)  # (1, 1, H, W)
    valid = torch.from_numpy(valid_np.astype(bool))
    errors = pred_error[0, 0][valid]
    metrics = compute_normal_metrics(errors)
    return errors, metrics


def save_pred_like_infer(pred_m11_chw: torch.Tensor, npy_path: Path, vis_path: Path | None):
    """Save like ``infer.py``: npy in [0, 1], optional uint8 vis png."""
    # pred is [-1,1] (1,3,H,W) from gen_normal; infer stores pipeline [0,1] HWC
    pred_01 = ((pred_m11_chw.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()) * 0.5 + 0.5).clip(0, 1)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, pred_01.astype(np.float32))
    if vis_path is not None:
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((pred_01 * 255).astype(np.uint8)).save(vis_path)


def resolve_pred_path(
    prediction_dir: Path, obj_name: str, light_stem: str | None = None
) -> Path:
    """Find a saved prediction for one object (and optional light stem)."""
    if light_stem is not None:
        candidates = (
            prediction_dir / obj_name / f"{light_stem}.npy",
            prediction_dir / "normal" / obj_name / f"{light_stem}.npy",
            prediction_dir / f"{obj_name}_{light_stem}.npy",
            prediction_dir / "normal" / f"{obj_name}_{light_stem}.npy",
            prediction_dir / obj_name / f"{light_stem}.png",
            prediction_dir / f"{obj_name}_{light_stem}.png",
        )
    else:
        candidates = (
            prediction_dir / f"{obj_name}.npy",
            prediction_dir / f"{obj_name}_norm.npy",
            prediction_dir / "normal" / f"{obj_name}.npy",
            prediction_dir / f"{obj_name}_norm.png",
            prediction_dir / f"{obj_name}.png",
            prediction_dir / "normal_vis" / f"{obj_name}.png",
        )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing prediction for {obj_name}"
        + (f" light={light_stem}" if light_stem else "")
        + f" under {prediction_dir}"
    )


def select_lit_images(obj_dir: Path, light_index: int | None) -> list[Path]:
    """All lit images, or a single index if ``light_index`` is set."""
    images = list_lit_images(obj_dir)
    if not images:
        raise FileNotFoundError(f"No lit images in {obj_dir}")
    if light_index is None:
        return images
    if light_index < 0 or light_index >= len(images):
        raise IndexError(
            f"light_index={light_index} out of range for {obj_dir.name} "
            f"({len(images)} images)"
        )
    return [images[light_index]]


# ========================= main =========================

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    seed_all(args.seed)

    image_root = resolve_image_root(args.data_dir, args.image_subdir)
    gt_root = resolve_gt_root(args.gt_dir, image_root)
    objects = discover_objects(image_root, args.objects)

    output_dir = Path(args.output_dir)
    # Mirror infer.py: normal/<OBJ>/<light_stem>.npy (+ optional vis)
    pred_npy_dir = output_dir / "normal"
    pred_npy_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "normal_vis" if args.save_vis else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA not available; running on CPU will be slow.")

    light_mode = (
        f"single light_index={args.light_index}"
        if args.light_index is not None
        else "ALL lights per object (~100)"
    )
    logging.info("Device: %s", device)
    logging.info("Image root: %s", image_root)
    logging.info("GT root: %s", gt_root)
    logging.info("Objects (%d): %s", len(objects), ", ".join(objects))
    logging.info("Light protocol: %s", light_mode)
    logging.info(
        "Object MAE = mean of per-light MAEs (each light vs same GT); "
        "scoring via normal_utils"
    )

    pipe = None
    if args.prediction_dir is None:
        pipe = build_pipeline(args, device)
        logging.info("Loaded pipeline: %s (%s)", args.pretrained_model_name_or_path, args.mode)
    else:
        logging.info("Scoring predictions from: %s", args.prediction_dir)

    per_object = {}
    all_errors = []  # pixel-pooled across all objects × lights

    for obj_name in tqdm(objects, desc="DiLiGenT-Pi objects"):
        image_dir = find_object_dir(image_root, obj_name)
        gt_dir = find_object_dir(gt_root, obj_name)
        gt_normal = load_gt_normal(gt_dir)
        mask = load_mask(image_dir)
        lit_images = select_lit_images(image_dir, args.light_index)

        obj_pred_dir = pred_npy_dir / obj_name
        obj_vis_dir = (vis_dir / obj_name) if vis_dir is not None else None
        if args.prediction_dir is None:
            obj_pred_dir.mkdir(parents=True, exist_ok=True)
            if obj_vis_dir is not None:
                obj_vis_dir.mkdir(parents=True, exist_ok=True)

        per_light = {}
        light_maes = []
        obj_errors = []

        for img_path in tqdm(lit_images, desc=f"{obj_name} lights", leave=False):
            light_stem = img_path.stem  # e.g. "001"

            if args.prediction_dir is None:
                rgb = load_rgb_like_infer(img_path, device)
                pred_bchw = gen_normal(rgb, pipe, args.timestep, args.processing_res)
                pred_hwc = (
                    pred_bchw.squeeze(0)
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                save_pred_like_infer(
                    pred_bchw,
                    obj_pred_dir / f"{light_stem}.npy",
                    (obj_vis_dir / f"{light_stem}.png") if obj_vis_dir is not None else None,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                pred_path = resolve_pred_path(
                    Path(args.prediction_dir), obj_name, light_stem=light_stem
                )
                pred_hwc = load_pred_like_score_finaldataset(pred_path)

            errors, metrics = score_normals(pred_hwc, gt_normal, mask)
            mae = float(metrics["mean"])
            light_maes.append(mae)
            obj_errors.append(errors.detach().cpu().reshape(-1))
            per_light[light_stem] = {
                "mae_deg": mae,
                "median_deg": float(metrics["median"]),
                "rmse_deg": float(metrics["rmse"]),
                "a_11_25": float(metrics["a3"]),
                "num_pixels": int(errors.numel()),
                "input_image": str(img_path),
            }

        # Object score = average of per-light MAEs (user / monocular-over-lights protocol)
        object_mae = float(np.mean(light_maes))
        obj_errors_t = torch.cat(obj_errors, dim=0)
        obj_pixel_metrics = compute_normal_metrics(obj_errors_t)

        per_object[obj_name] = {
            "mae_deg": object_mae,  # mean of per-light MAEs
            "num_lights": len(light_maes),
            "per_light_mae_deg": {k: v["mae_deg"] for k, v in per_light.items()},
            "median_deg": float(obj_pixel_metrics["median"]),
            "rmse_deg": float(obj_pixel_metrics["rmse"]),
            "a_11_25": float(obj_pixel_metrics["a3"]),
            "pixel_pooled_mae_deg": float(obj_pixel_metrics["mean"]),
            "num_pixels": int(obj_errors_t.numel()),
            "per_light": per_light,
        }
        all_errors.append(obj_errors_t)
        logging.info(
            "%s: MAE=%.3f° (mean of %d lights)  pixel-pooled=%.3f°",
            obj_name,
            object_mae,
            len(light_maes),
            float(obj_pixel_metrics["mean"]),
        )

    all_errors_t = torch.cat(all_errors, dim=0)
    global_metrics = compute_normal_metrics(all_errors_t)

    # Overall Avg = mean of per-object MAEs (each object MAE already mean-over-lights)
    object_maes = [per_object[o]["mae_deg"] for o in objects if o in per_object]
    object_mean_mae = float(np.mean(object_maes)) if object_maes else float("nan")

    group_mae = {}
    for group_name, group_objs in MATERIAL_GROUPS.items():
        vals = [per_object[o]["mae_deg"] for o in group_objs if o in per_object]
        if vals:
            group_mae[group_name] = float(np.mean(vals))

    summary = {
        "dataset": "DiLiGenT-Pi",
        "protocol": (
            "For each object: predict normals for every lit image, score each vs the "
            "same Normal_gt (Lotus normal_utils), object MAE = mean of per-light MAEs. "
            "Overall Avg = mean of object MAEs."
        ),
        "metric": "MAngE = Lotus normal_utils 'mean' (degrees)",
        "metric_source": (
            "evaluation.util.normal_utils.compute_normal_error + compute_normal_metrics "
            "(same as score_finaldataset.py / evaluation_normal)"
        ),
        "forward_source": "eval.py gen_normal (processing_res default 0)",
        "num_objects": len(per_object),
        "light_index": args.light_index,  # None => all lights
        "lights_per_object": {
            o: per_object[o]["num_lights"] for o in per_object
        },
        "image_root": str(image_root),
        "gt_root": str(gt_root),
        "model": (
            args.pretrained_model_name_or_path
            if args.prediction_dir is None
            else f"predictions:{args.prediction_dir}"
        ),
        "mode": args.mode,
        "processing_res": args.processing_res,
        # Pixel-pooled over all objects × all lights
        "global_mae_deg": float(global_metrics["mean"]),
        "global_median_deg": float(global_metrics["median"]),
        "global_rmse_deg": float(global_metrics["rmse"]),
        "global_a_11_25": float(global_metrics["a3"]),
        # Primary reported numbers
        "object_mean_mae_deg": object_mean_mae,
        "group_mean_mae_deg": group_mae,
        "per_object_mae_deg": {k: v["mae_deg"] for k, v in per_object.items()},
        "per_object": per_object,
    }

    json_path = output_dir / "eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    table_rows = [
        [
            obj,
            f"{per_object[obj]['num_lights']}",
            f"{per_object[obj]['mae_deg']:.3f}",
        ]
        for obj in objects
        if obj in per_object
    ]
    table_rows.append(["Avg (mean of objects)", "", f"{object_mean_mae:.3f}"])
    table_rows.append(["GLOBAL (pixel-pooled all lights)", "", f"{summary['global_mae_deg']:.3f}"])
    for g, v in group_mae.items():
        table_rows.append([f"{g} avg", "", f"{v:.3f}"])

    txt = tabulate(
        table_rows,
        headers=["Object", "#lights", "MAE (deg)"],
        tablefmt="github",
    )
    txt_path = output_dir / "eval_metrics.txt"
    with open(txt_path, "w") as f:
        f.write("# Per-object MAE = mean of per-light MAEs (each light vs same GT)\n")
        f.write(txt + "\n")

    print("\n=== Per-object MAE (mean over lights) ===")
    print(txt)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")
    print("\nPer-object MAE (°):")
    for obj in objects:
        if obj in per_object:
            print(f"  {obj:12s}  {per_object[obj]['mae_deg']:.3f}")
    print(f"  {'Avg':12s}  {object_mean_mae:.3f}")


if __name__ == "__main__":
    main()
