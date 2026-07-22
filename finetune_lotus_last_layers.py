#!/usr/bin/env python
"""Fine-tune only the last few UNet layers of Lotus (G or D) on finaldataset.

Loads a pretrained Lotus normal checkpoint, freezes VAE + text encoder + early UNet,
trains the last N up-blocks + conv_norm_out / conv_out on disc RGB/normal/mask triples.

Example (Lotus G):
  CUDA_VISIBLE_DEVICES=0 python finetune_lotus_last_layers.py \\
    --mode generation \\
    --pretrained_model_name_or_path jingheya/lotus-normal-g-v1-1 \\
    --train_data_dir /path/to/finaldataset \\
    --output_dir output/finetune_lotus_g_discs \\
    --max_train_steps 1000 --train_batch_size 1 --resolution 576

Example (Lotus D):
  CUDA_VISIBLE_DEVICES=1 python finetune_lotus_last_layers.py \\
    --mode regression \\
    --pretrained_model_name_or_path jingheya/lotus-normal-d-v1-1 \\
    --train_data_dir /path/to/finaldataset \\
    --output_dir output/finetune_lotus_d_discs \\
    --max_train_steps 1000 --train_batch_size 1 --resolution 576

After training, run infer.py with --pretrained_model_name_or_path pointing at output_dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from evaluation.util.evaluation_normal_score import (
    accumulate_normal_errors,
    metrics_from_errors,
)
from pipeline import LotusDPipeline, LotusGPipeline

check_min_version("0.28.0.dev0")
logger = get_logger(__name__, log_level="INFO")


class FinalDatasetDiscs(Dataset):
    """finaldataset/{stem}/rgb.png + normal_map.png + mask.png"""

    def __init__(
        self,
        data_dir: str,
        resolution: int = 576,
        random_flip: bool = True,
        include_syn: bool = True,
        seed: int = 42,
        stem_allowlist: list[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.random_flip = random_flip
        self.rng = random.Random(seed)

        if stem_allowlist is not None:
            stems = list(stem_allowlist)
        else:
            stems = sorted(
                p.name for p in self.data_dir.iterdir() if p.is_dir() and (p / "rgb.png").is_file()
            )
        if not include_syn:
            stems = [s for s in stems if "_syn_" not in s]

        self.samples = []
        for stem in stems:
            d = self.data_dir / stem
            rgb = d / "rgb.png"
            normal = d / "normal_map.png"
            mask = d / "mask.png"
            if rgb.is_file() and normal.is_file() and mask.is_file():
                self.samples.append((stem, rgb, normal, mask))

        if not self.samples:
            raise FileNotFoundError(f"No valid samples under {data_dir}")

    def __len__(self):
        return len(self.samples)

    def _resize_square(self, img: Image.Image, resample) -> Image.Image:
        return img.resize((self.resolution, self.resolution), resample)

    def __getitem__(self, idx):
        stem, rgb_path, normal_path, mask_path = self.samples[idx]
        rgb = Image.open(rgb_path).convert("RGB")
        normal = Image.open(normal_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Match Lotus VKITTI: bilinear RGB, nearest normals/masks
        rgb = self._resize_square(rgb, Image.BILINEAR)
        normal = self._resize_square(normal, Image.NEAREST)
        mask = self._resize_square(mask, Image.NEAREST)

        if self.random_flip and self.rng.random() < 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            normal = normal.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # Same x-flip as utils/vkitti_dataset.py VKITTITransform
            n = np.array(normal)
            normal_mask = np.any(n != [0, 0, 0], axis=-1)
            n[:, :, 0][normal_mask] = 255 - n[:, :, 0][normal_mask]
            normal = Image.fromarray(n)

        # Same [-1, 1] encoding as Lotus train (vkitti / hypersim)
        rgb_t = torch.from_numpy(np.array(rgb).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        normal_t = torch.from_numpy(np.array(normal).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        mask_t = torch.from_numpy((np.array(mask).astype(np.float32) > 127).astype(np.float32))[None]

        return {
            "pixel_values": rgb_t.contiguous().float(),
            "normal_values": normal_t.contiguous().float(),
            "valid_mask_values": mask_t.contiguous().float(),
            "stem": stem,
        }


def collate_fn(examples):
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
        "normal_values": torch.stack([e["normal_values"] for e in examples]),
        "valid_mask_values": torch.stack([e["valid_mask_values"] for e in examples]),
        "stem": [e["stem"] for e in examples],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune last Lotus UNet layers on discs.")
    p.add_argument(
        "--mode",
        type=str,
        choices=["generation", "regression"],
        required=True,
        help="generation = Lotus G; regression = Lotus D",
    )
    p.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="HF id or local dir, e.g. jingheya/lotus-normal-g-v1-1",
    )
    p.add_argument("--train_data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resolution", type=int, default=576)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--lr_scheduler", type=str, default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--checkpoints_total_limit", type=int, default=2)
    p.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Every N steps: disc val on val_stems (stock Lotus mean). "
        "Keeps checkpoint-best. 0 disables.",
    )
    p.add_argument(
        "--random_flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Horizontal flip aug (default on). Use --no-random_flip to disable.",
    )
    p.add_argument("--allow_tf32", action="store_true", help="TF32 matmul on Ampere+ GPUs.")
    p.add_argument(
        "--exclude_syn",
        action="store_true",
        help="Train on GT-only stems (no _syn_). Default uses GT+syn in the train split.",
    )
    p.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Dir with train_stems.txt (+ val/test lists) from make_finaldataset_split.py. "
        "If set, trains ONLY on train_stems.txt (tune on val; final report on test).",
    )
    p.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help="If set (and --split_dir not given), create a stratified split in output_dir/split/.",
    )
    p.add_argument(
        "--train_last_n_up_blocks",
        type=int,
        default=1,
        help="Used when --trainable_scope=last_up_blocks. Unfreeze last N up_blocks + head.",
    )
    p.add_argument(
        "--trainable_scope",
        type=str,
        default="last_resnet",
        choices=["head", "last_resnet", "last_up_blocks"],
        help="How much of the UNet to unfreeze (least → most capacity): "
        "head ≈ 0.01M; last_resnet ≈ 3.4M (recommended light FT); "
        "last_up_blocks ≈ 19M (previous default, overfits discs easily).",
    )
    p.add_argument(
        "--also_train_conv_in",
        action="store_true",
        help="Also unfreeze conv_in (usually leave frozen).",
    )
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    return p.parse_args()


def load_stem_list(path: Path) -> list[str]:
    stems = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not stems:
        raise ValueError(f"Empty stem list: {path}")
    return stems


def resolve_train_stems(args) -> tuple[list[str] | None, Path | None, list[str] | None]:
    """Return (train_stems or None=all, split_dir, val_stems or None)."""
    if args.split_dir:
        split_dir = Path(args.split_dir)
        train_path = split_dir / "train_stems.txt"
        if not train_path.is_file():
            raise FileNotFoundError(f"Missing {train_path}. Run make_finaldataset_split.py first.")
        val_path = split_dir / "val_stems.txt"
        val_stems = load_stem_list(val_path) if val_path.is_file() else None
        return load_stem_list(train_path), split_dir, val_stems

    if args.val_ratio is not None:
        from make_finaldataset_split import list_stems, stratified_split, write_list

        split_dir = Path(args.output_dir) / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        stems = list_stems(Path(args.train_data_dir))
        parts = stratified_split(stems, args.val_ratio, args.seed)
        write_list(split_dir / "train_stems.txt", parts["train"])
        write_list(split_dir / "val_stems.txt", parts["val"])
        info = {
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "counts": {k: len(v) for k, v in parts.items()},
            "protocol": "Train on train_stems; score val_stems with all/gt_only/syn_only.",
        }
        (split_dir / "split_info.json").write_text(json.dumps(info, indent=2) + "\n")
        logger.info("Created split at %s: train=%d val=%d", split_dir, len(parts["train"]), len(parts["val"]))
        return parts["train"], split_dir, parts["val"]

    return None, None, None


def gen_normal(img, pipe, prompt="", timestep=999):
    """Copied from train_lotus_d.run_evaluation nested gen_normal."""
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(pipe.device.type)

    with autocast_ctx:
        task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(pipe.device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

        pred_normal = pipe(
            rgb_in=img,  # [-1,1]
            task_emb=task_emb,
            prompt=prompt,
            timesteps=[timestep],
            output_type="pt",
        ).images[0]  # [0,1], (3,h,w)
        pred_normal = (pred_normal * 2 - 1.0).unsqueeze(0)  # [-1,1], (1,3,h,w)
    return pred_normal


def build_val_pipeline(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype):
    """Same construction as train_lotus_d.log_validation (G or D by --mode)."""
    pipe_cls = LotusGPipeline if args.mode == "generation" else LotusDPipeline
    pipeline = pipe_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    return pipeline


def _load_rgb_for_val(rgb_path: Path, device):
    """Same RGB preprocess as train_lotus_d.run_example_validation (normal)."""
    validation_image = Image.open(rgb_path).convert("RGB")
    validation_image = np.array(validation_image).astype(np.float32)
    validation_image = torch.tensor(validation_image).permute(2, 0, 1).unsqueeze(0)
    validation_image = validation_image / 127.5 - 1.0
    return validation_image.to(device)


def _load_gt_mask(gt_path: Path, mask_path: Path, device):
    """Disc GT → tensors shaped like evaluation_normal's gt_norm / gt_norm_mask."""
    gt = cv2.cvtColor(cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    gt = gt.astype(np.float32)
    if gt.max() > 1.5:
        gt = gt / 255.0 * 2.0 - 1.0
    gt_t = (
        torch.from_numpy(np.ascontiguousarray(gt))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )  # (1, 3, H, W)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
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
    # pred_error is (B,1,H,W); index like evaluation_normal: pred_error[gt_norm_mask]
    gt_norm_mask = torch.from_numpy(mask.astype(bool)).to(device)[None, None]  # (1,1,H,W)
    return gt_t, gt_norm_mask


def run_disc_evaluation(pipeline, data_dir: Path, val_stems: list[str], timestep: int, device):
    """Disc data loop; scoring via shared evaluation_normal_score (stock accumulate)."""
    from torchvision.transforms.functional import resize

    total_normal_errors = None

    for stem in tqdm(val_stems, desc="val", leave=False):
        sample_dir = data_dir / stem
        rgb_path = sample_dir / "rgb.png"
        gt_path = sample_dir / "normal_map.png"
        mask_path = sample_dir / "mask.png"
        if not (rgb_path.is_file() and gt_path.is_file() and mask_path.is_file()):
            continue

        rgb = _load_rgb_for_val(rgb_path, device)
        with torch.no_grad():
            pred_norm = gen_normal(rgb, pipeline, prompt="", timestep=timestep)

        gt_norm, gt_norm_mask = _load_gt_mask(gt_path, mask_path, device)
        if gt_norm is None:
            continue

        if pred_norm.shape[-2:] != gt_norm.shape[-2:]:
            pred_norm = resize(pred_norm, gt_norm.shape[-2:], antialias=True)

        total_normal_errors = accumulate_normal_errors(
            pred_norm, gt_norm, gt_norm_mask, total_normal_errors
        )

    metrics = metrics_from_errors(total_normal_errors)
    if metrics is None:
        return {"mean": float("inf"), "median": float("nan"), "a3": float("nan")}
    return metrics


def log_disc_validation(
    vae,
    text_encoder,
    tokenizer,
    unet,
    args,
    accelerator,
    weight_dtype,
    step,
    val_stems: list[str],
    best_state: dict,
):
    """Same shape as train_lotus_d.log_validation: build pipe → evaluate → cleanup."""
    logger.info("Running disc validation at step %d (%d stems)...", step, len(val_stems))

    unet_model = accelerator.unwrap_model(unet)
    was_training = unet_model.training
    unet_model.eval()

    pipeline = build_val_pipeline(
        vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype
    )

    metrics = run_disc_evaluation(
        pipeline,
        Path(args.train_data_dir),
        val_stems,
        timestep=args.timestep,
        device=accelerator.device,
    )

    # same leader as train_lotus_d TOP5_STEPS_NORMAL
    mean_value = metrics["mean"] if metrics["mean"] == metrics["mean"] else float("inf")

    logger.info(
        "Val step-%d | mean=%.3f median=%.3f a3(<11.25)=%.3f",
        step,
        metrics.get("mean", float("nan")),
        metrics.get("median", float("nan")),
        metrics.get("a3", float("nan")),
    )
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_scalar("val/mean", metrics["mean"], step)
            tracker.writer.add_scalar("val/11.25", metrics.get("a3", float("nan")), step)

    with open(os.path.join(args.output_dir, "val_history.jsonl"), "a") as f:
        f.write(json.dumps({"step": step, **{k: float(v) for k, v in metrics.items()}}) + "\n")

    if mean_value < best_state["mean"]:
        best_state["mean"] = mean_value
        best_state["step"] = step
        best_dir = os.path.join(args.output_dir, "checkpoint-best")
        os.makedirs(best_dir, exist_ok=True)
        unet_model.save_pretrained(os.path.join(best_dir, "unet"))
        with open(os.path.join(best_dir, "best_val.json"), "w") as f:
            json.dump(
                {
                    "step": step,
                    "mean": float(mean_value),
                    "metrics": {k: float(v) for k, v in metrics.items()},
                },
                f,
                indent=2,
            )
        logger.info("New best val mean=%.3f at step %d → %s", mean_value, step, best_dir)

    del pipeline
    torch.cuda.empty_cache()
    if was_training:
        unet_model.train()


def freeze_unet_except_last_layers(
    unet,
    last_n_up_blocks: int = 1,
    also_conv_in: bool = False,
    trainable_scope: str = "last_resnet",
):
    """Freeze UNet; unfreeze a small output-side scope to limit disc overfitting.

    Scopes (SD2 UNet ≈ 868M params total):
      head          — conv_norm_out + conv_out only (~0.01M)
      last_resnet   — up_blocks.3.resnets.2 + head (~3.4M)  [light FT]
      last_up_blocks— last N up_blocks + head (~19M if N=1) [previous; overfits]
    """
    n_up = len(unet.up_blocks)
    if last_n_up_blocks < 1 or last_n_up_blocks > n_up:
        raise ValueError(f"train_last_n_up_blocks must be in [1, {n_up}], got {last_n_up_blocks}")

    last_idx = n_up - 1  # up_blocks.3 on SD2
    if trainable_scope == "head":
        trainable_prefixes = ["conv_norm_out.", "conv_out."]
    elif trainable_scope == "last_resnet":
        # Final ResNet in the last up-block only (not its attentions / earlier resnets)
        trainable_prefixes = [
            f"up_blocks.{last_idx}.resnets.2.",
            "conv_norm_out.",
            "conv_out.",
        ]
    elif trainable_scope == "last_up_blocks":
        start_idx = n_up - last_n_up_blocks
        trainable_prefixes = [f"up_blocks.{i}." for i in range(start_idx, n_up)]
        trainable_prefixes += ["conv_norm_out.", "conv_out."]
    else:
        raise ValueError(f"Unknown trainable_scope: {trainable_scope}")

    if also_conv_in:
        trainable_prefixes.append("conv_in.")

    unet.requires_grad_(False)
    trainable_names = []
    for name, param in unet.named_parameters():
        if any(name.startswith(pref) for pref in trainable_prefixes):
            param.requires_grad_(True)
            trainable_names.append(name)

    total = sum(p.numel() for p in unet.parameters())
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(
        "UNet freeze scope=%s: trainable %s / %s params (%.2f%%). prefixes=%s",
        trainable_scope,
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
        trainable_prefixes,
    )
    if trainable == 0:
        raise RuntimeError(f"No params matched scope={trainable_scope} prefixes={trainable_prefixes}")
    return trainable_names


def task_embeddings(device, bsz_per_task: int):
    task_emb_anno = torch.tensor([1, 0], device=device, dtype=torch.float32).unsqueeze(0)
    task_emb_anno = torch.cat([torch.sin(task_emb_anno), torch.cos(task_emb_anno)], dim=-1).repeat(
        bsz_per_task, 1
    )
    task_emb_rgb = torch.tensor([0, 1], device=device, dtype=torch.float32).unsqueeze(0)
    task_emb_rgb = torch.cat([torch.sin(task_emb_rgb), torch.cos(task_emb_rgb)], dim=-1).repeat(
        bsz_per_task, 1
    )
    return torch.cat((task_emb_anno, task_emb_rgb), dim=0)


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=str(logging_dir)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "finetune_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    train_stems, split_dir, val_stems = resolve_train_stems(args)
    if split_dir is not None and accelerator.is_main_process:
        logger.info(
            "Using split_dir=%s (train; val→best ckpt; test via score_finaldataset --split_dir)",
            split_dir,
        )

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = None
    if args.mode == "generation":
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    freeze_unet_except_last_layers(
        unet,
        last_n_up_blocks=args.train_last_n_up_blocks,
        also_conv_in=args.also_train_conv_in,
        trainable_scope=args.trainable_scope,
    )
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters — check --train_last_n_up_blocks")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = FinalDatasetDiscs(
        args.train_data_dir,
        resolution=args.resolution,
        random_flip=args.random_flip,
        include_syn=not args.exclude_syn,
        seed=args.seed,
        stem_allowlist=train_stems,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True if len(dataset) >= args.train_batch_size else False,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Empty prompt embedding (Lotus convention)
    text_inputs = tokenizer(
        "",
        padding="do_not_pad",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        empty_emb = text_encoder(
            text_inputs.input_ids.to(accelerator.device), return_dict=False
        )[0]

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / max(num_update_steps_per_epoch, 1))

    if accelerator.is_main_process:
        accelerator.init_trackers("finetune_lotus_last_layers", config=vars(args))

    logger.info("***** Running last-layer fine-tune *****")
    logger.info(f"  Mode = {args.mode} ({'Lotus G' if args.mode == 'generation' else 'Lotus D'})")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num epochs (approx) = {num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Resolution = {args.resolution}")
    logger.info(f"  Trainable scope = {args.trainable_scope}")
    logger.info(f"  Last N up_blocks (if scope=last_up_blocks) = {args.train_last_n_up_blocks}")
    logger.info(f"  Output = {args.output_dir}")
    if split_dir is not None:
        logger.info(f"  Split dir = {split_dir}")
        logger.info(
            "  Protocol: val_stems for checkpoint-best; "
            "score_finaldataset.py --split_dir for test_stems"
        )

    do_validation = (
        args.validation_steps > 0 and val_stems is not None and len(val_stems) > 0
    )
    if do_validation:
        logger.info(
            "  Disc validation every %d steps (%d val stems)",
            args.validation_steps,
            len(val_stems),
        )
    best_state = {"mean": float("inf"), "step": None}

    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    if do_validation and accelerator.is_main_process:
        log_disc_validation(
            vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype,
            0, val_stems, best_state,
        )
    accelerator.wait_for_everyone()

    for epoch in range(num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Same batch keys / encoding path as train_lotus_{g,d}.py
                rgb = batch["pixel_values"].to(dtype=weight_dtype)
                normal = batch["normal_values"].to(dtype=weight_dtype)
                valid_mask = batch["valid_mask_values"].to(accelerator.device)

                # Dual-task batch: [normals..., rgbs...] — identical to stock Lotus
                with torch.no_grad():
                    rgb_latents = vae.encode(torch.cat([rgb, rgb], dim=0)).latent_dist.sample()
                    rgb_latents = rgb_latents * vae.config.scaling_factor
                    target_latents = vae.encode(torch.cat([normal, rgb], dim=0)).latent_dist.sample()
                    target_latents = target_latents * vae.config.scaling_factor

                bsz = target_latents.shape[0]
                bsz_per_task = bsz // 2
 
                invalid_mask = ~valid_mask.bool()
                valid_mask_down_anno = ~torch.max_pool2d(invalid_mask.float(), 8, 8).bool()
                valid_mask_down_anno = valid_mask_down_anno.repeat(1, 4, 1, 1)
                valid_mask_down_rgb = torch.ones_like(target_latents[bsz_per_task:]).to(target_latents.device).bool()

                timesteps = torch.tensor([args.timestep], device=target_latents.device).repeat(bsz).long()

                if args.mode == "generation":
                    # Lotus G: concat RGB latents + noisy target latents (8-ch)
                    noise = torch.randn_like(target_latents)
                    noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                    unet_input = torch.cat([rgb_latents, noisy_latents], dim=1)
                else:
                    # Lotus D: RGB latents only (4-ch)
                    unet_input = rgb_latents

                encoder_hidden_states = empty_emb.to(dtype=weight_dtype).repeat(bsz, 1, 1)
                task_emb = task_embeddings(accelerator.device, bsz_per_task)
                target = target_latents

                model_pred = unet(
                    unet_input,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                    class_labels=task_emb,
                )[0]

                # Same masked MSE as stock Lotus (prediction_type=sample → target latents)
                anno_loss = F.mse_loss(
                    model_pred[:bsz_per_task][valid_mask_down_anno].float(),
                    target[:bsz_per_task][valid_mask_down_anno].float(),
                    reduction="mean",
                )
                rgb_loss = F.mse_loss(
                    model_pred[bsz_per_task:][valid_mask_down_rgb].float(),
                    target[bsz_per_task:][valid_mask_down_rgb].float(),
                    reduction="mean",
                )

                loss = anno_loss + rgb_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in unet.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # Same postfix keys as stock Lotus train logs
                logs = {
                    "SL": loss.detach().item(),
                    "SL_A": anno_loss.detach().item(),
                    "SL_R": rgb_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(unet)
                    unwrapped.save_pretrained(os.path.join(ckpt_dir, "unet"))
                    logger.info(f"Saved UNet checkpoint to {ckpt_dir}")

                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [
                                d
                                for d in os.listdir(args.output_dir)
                                if d.startswith("checkpoint-") and d[len("checkpoint-"):].isdigit()
                            ],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        while len(ckpts) > args.checkpoints_total_limit:
                            old = ckpts.pop(0)
                            shutil.rmtree(os.path.join(args.output_dir, old), ignore_errors=True)

                if (
                    do_validation
                    and accelerator.is_main_process
                    and global_step % args.validation_steps == 0
                ):
                    log_disc_validation(
                        vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype,
                        global_step, val_stems, best_state,
                    )

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        best_unet = os.path.join(args.output_dir, "checkpoint-best", "unet")
        if best_state["step"] is not None and os.path.isdir(best_unet):
            logger.info(
                "Loading best-val UNet from step %s (val mean=%.3f)",
                best_state["step"],
                best_state["mean"],
            )
            unet = UNet2DConditionModel.from_pretrained(best_unet)
            unet.to(accelerator.device, dtype=weight_dtype)

        pipe_cls = LotusGPipeline if args.mode == "generation" else LotusDPipeline
        pipeline = pipe_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(args.output_dir)
        logger.info(f"Saved full pipeline to {args.output_dir}")
        if split_dir is not None:
            logger.info(
                "Final metrics: score_finaldataset.py --split_dir %s",
                split_dir,
            )
        logger.info(
            "Infer with: python infer.py --pretrained_model_name_or_path %s "
            "--mode %s --task_name normal ...",
            args.output_dir,
            args.mode,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
