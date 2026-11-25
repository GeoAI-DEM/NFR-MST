#!/usr/bin/env python3
"""
Train Mamba+Swin regressor on pre-cut patch datasets.

Supports:
    - single GPU
    - CPU
    - single-process multi-GPU with DataParallel
    - single-node multi-GPU with torchrun / DDP

Recommended launch:
    Single GPU:
        python train_mamba_swin.py

    Dual GPU with config MODE=dp:
        python train_mamba_swin.py

    Dual GPU with DDP:
        torchrun --nproc_per_node=2 train_mamba_swin.py
"""

from __future__ import annotations

import csv
from contextlib import nullcontext
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import MambaSwinRegressor, MambaSwinRegressorConfig


# =========================
# CONFIG
# =========================

TRAIN_PATCH_H5 = PROJECT_ROOT / "train_patches.h5"
VAL_PATCH_H5 = PROJECT_ROOT / "val_patches.h5"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "run_fuzzy_gate_only"
CHECKPOINT_DIRNAME = "checkpoints"
BEST_DIRNAME = "best"
BEST_FILENAME = "best.pt"

TRAIN_MODE = "ddp"  # "single", "dp", "ddp"
GPU_IDS = [0, 1]
PRELOAD_PATCHES_TO_MEMORY = True

SEED = 20260401
EPOCHS = 100
BATCH_SIZE_PER_GPU = 256
NUM_WORKERS = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

LEARNING_RATE = 8.0e-5
WEIGHT_DECAY = 1.0e-4
BETAS = (0.9, 0.999)
GRAD_CLIP_NORM = 1.0
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 5
PLATEAU_THRESHOLD = 1.0e-3
MIN_LR = 1.0e-6

USE_AMP = True
AMP_DTYPE = "bf16"  # "fp16" or "bf16"

LOSS_NAME = "mse"  # "huber", "mse", "l1"
HUBER_DELTA = 1.0

USE_FUZZY_LOGIC = True
FUZZY_USE_LOSS_WEIGHT = False
FUZZY_USE_FEATURE_GATE = True
FUZZY_USE_N_POINTS = True
FUZZY_USE_RESIDUAL_STD = True
FUZZY_WEIGHT_MODE = "online"  # "online" or "precomputed"
FUZZY_WEIGHT_EPS = 1.0e-8

SAVE_EVERY_EPOCH = True

MODEL_CFG = MambaSwinRegressorConfig(
    patch_size=33,
    in_chans=6,
    embed_dim=64,
    swin_depths=(4, 4),
    swin_num_heads=(4, 4),
    window_size=11,
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.05,
    mamba_layers=4,
    mamba_d_state=16,
    mamba_d_conv=4,
    mamba_expand=2,
    head_hidden_dim=128,
    fuzzy_enabled=USE_FUZZY_LOGIC,
    fuzzy_use_loss_weight=USE_FUZZY_LOGIC and FUZZY_USE_LOSS_WEIGHT,
    fuzzy_use_feature_gate=USE_FUZZY_LOGIC and FUZZY_USE_FEATURE_GATE,
    fuzzy_use_n_points=USE_FUZZY_LOGIC and FUZZY_USE_N_POINTS,
    fuzzy_use_residual_std=USE_FUZZY_LOGIC and FUZZY_USE_RESIDUAL_STD,
    fuzzy_hidden_dim=16,
)


def is_dist_enabled() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[bool, int, int, int]:
    if TRAIN_MODE.lower() != "ddp" or not is_dist_enabled():
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def is_parallel_model(model: nn.Module) -> bool:
    return isinstance(model, (DDP, nn.DataParallel))


def get_autocast_dtype() -> torch.dtype:
    if AMP_DTYPE.lower() == "bf16":
        return torch.bfloat16
    return torch.float16


def assert_finite_tensor(tensor: torch.Tensor, name: str, epoch: int, step_idx: int) -> None:
    if torch.isfinite(tensor).all():
        return
    raise RuntimeError(
        f"Non-finite {name} detected at epoch={epoch}, step={step_idx}."
    )


def assert_finite_gradients(model: nn.Module, epoch: int, step_idx: int) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise RuntimeError(
                f"Non-finite gradient detected at epoch={epoch}, step={step_idx}, parameter={name}."
            )


class PatchH5Dataset(Dataset):
    def __init__(self, h5_path: Path) -> None:
        self.h5_path = h5_path
        self._file = None
        self.preloaded = PRELOAD_PATCHES_TO_MEMORY
        self._x = None
        self._y = None
        self._row = None
        self._col = None
        self._residual_std = None
        self._n_points = None
        self._conf = None
        self._weight = None
        self._val_block_id = None

        with h5py.File(h5_path, "r") as f:
            self.length = int(f["x"].shape[0])
            if self.preloaded:
                self._x = f["x"][:].astype(np.float32, copy=False)
                self._y = f["y"][:].astype(np.float32, copy=False)
                self._row = f["row"][:].astype(np.int64, copy=False)
                self._col = f["col"][:].astype(np.int64, copy=False)
                self._residual_std = f["residual_std"][:].astype(np.float32, copy=False)
                self._n_points = f["n_points"][:].astype(np.int64, copy=False)
                self._conf = f["conf"][:].astype(np.float32, copy=False)
                self._weight = f["weight"][:].astype(np.float32, copy=False)
                self._val_block_id = f["val_block_id"][:].astype(np.int64, copy=False)

    def __len__(self) -> int:
        return self.length

    def _ensure_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.preloaded:
            x = self._x[idx]
            y = self._y[idx]
            row = self._row[idx]
            col = self._col[idx]
            residual_std = self._residual_std[idx]
            n_points = self._n_points[idx]
            conf = self._conf[idx]
            weight = self._weight[idx]
            val_block_id = self._val_block_id[idx]
        else:
            f = self._ensure_file()
            x = f["x"][idx].astype(np.float32)
            y = np.float32(f["y"][idx])
            row = np.int64(f["row"][idx])
            col = np.int64(f["col"][idx])
            residual_std = np.float32(f["residual_std"][idx])
            n_points = np.int64(f["n_points"][idx])
            conf = np.float32(f["conf"][idx])
            weight = np.float32(f["weight"][idx])
            val_block_id = np.int64(f["val_block_id"][idx])

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "row": torch.tensor(row, dtype=torch.int64),
            "col": torch.tensor(col, dtype=torch.int64),
            "residual_std": torch.tensor(residual_std, dtype=torch.float32),
            "n_points": torch.tensor(n_points, dtype=torch.int64),
            "conf": torch.tensor(conf, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "val_block_id": torch.tensor(val_block_id, dtype=torch.int64),
        }


def create_loss() -> nn.Module:
    name = LOSS_NAME.lower()
    if name == "mse":
        return nn.MSELoss(reduction="none")
    if name == "l1":
        return nn.L1Loss(reduction="none")
    return nn.HuberLoss(delta=HUBER_DELTA, reduction="none")


def create_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )


def create_scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
        threshold=PLATEAU_THRESHOLD,
        min_lr=MIN_LR,
    )


def reduce_scalar(value: float, device: torch.device, world_size: int) -> float:
    if world_size == 1:
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / world_size)


def reduce_sum(value: float, device: torch.device, world_size: int) -> float:
    if world_size == 1:
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    start_time = time.time()
    steps_per_epoch = len(loader)

    iterator = loader
    if is_main_process(rank):
        iterator = tqdm(
            loader,
            total=steps_per_epoch,
            desc=f"Epoch {epoch:03d} train",
            leave=True,
        )

    for step_idx, batch in enumerate(iterator, start=1):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        conf = batch["conf"].to(device, non_blocking=True)
        n_points = batch["n_points"].to(device, non_blocking=True)
        residual_std = batch["residual_std"].to(device, non_blocking=True)
        precomputed_weight = batch["weight"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=get_autocast_dtype())
            if USE_AMP and device.type == "cuda"
            else nullcontext()
        )

        if scaler is not None:
            with autocast_ctx:
                model_out = (
                    model(
                        x,
                        conf=conf,
                        n_points=n_points,
                        residual_std=residual_std,
                        return_aux=USE_FUZZY_LOGIC,
                    )
                    if USE_FUZZY_LOGIC
                    else model(x)
                )
                if isinstance(model_out, dict):
                    pred = model_out["pred"]
                    fuzzy_weight = model_out.get("fuzzy_weight")
                else:
                    pred = model_out
                    fuzzy_weight = None
                loss_vec = criterion(pred, y)
                if USE_FUZZY_LOGIC and FUZZY_USE_LOSS_WEIGHT:
                    sample_weight = fuzzy_weight if FUZZY_WEIGHT_MODE == "online" else precomputed_weight
                    sample_weight = sample_weight.to(dtype=loss_vec.dtype)
                    loss = torch.sum(sample_weight * loss_vec) / (torch.sum(sample_weight) + FUZZY_WEIGHT_EPS)
                else:
                    loss = loss_vec.mean()
            assert_finite_tensor(pred, "prediction", epoch, step_idx)
            if USE_FUZZY_LOGIC and FUZZY_USE_LOSS_WEIGHT:
                assert_finite_tensor(sample_weight.detach(), "fuzzy_weight", epoch, step_idx)
            assert_finite_tensor(loss_vec.detach(), "loss_vector", epoch, step_idx)
            assert_finite_tensor(loss.detach(), "loss", epoch, step_idx)
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            assert_finite_gradients(model, epoch, step_idx)
            scaler.step(optimizer)
            scaler.update()
        else:
            with autocast_ctx:
                model_out = (
                    model(
                        x,
                        conf=conf,
                        n_points=n_points,
                        residual_std=residual_std,
                        return_aux=USE_FUZZY_LOGIC,
                    )
                    if USE_FUZZY_LOGIC
                    else model(x)
                )
                if isinstance(model_out, dict):
                    pred = model_out["pred"]
                    fuzzy_weight = model_out.get("fuzzy_weight")
                else:
                    pred = model_out
                    fuzzy_weight = None
                loss_vec = criterion(pred, y)
                if USE_FUZZY_LOGIC and FUZZY_USE_LOSS_WEIGHT:
                    sample_weight = fuzzy_weight if FUZZY_WEIGHT_MODE == "online" else precomputed_weight
                    sample_weight = sample_weight.to(dtype=loss_vec.dtype)
                    loss = torch.sum(sample_weight * loss_vec) / (torch.sum(sample_weight) + FUZZY_WEIGHT_EPS)
                else:
                    loss = loss_vec.mean()
            assert_finite_tensor(pred, "prediction", epoch, step_idx)
            if USE_FUZZY_LOGIC and FUZZY_USE_LOSS_WEIGHT:
                assert_finite_tensor(sample_weight.detach(), "fuzzy_weight", epoch, step_idx)
            assert_finite_tensor(loss_vec.detach(), "loss_vector", epoch, step_idx)
            assert_finite_tensor(loss.detach(), "loss", epoch, step_idx)
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            assert_finite_gradients(model, epoch, step_idx)
            optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_count += batch_size

        if is_main_process(rank):
            running_loss = total_loss / max(total_count, 1)
            if hasattr(iterator, "set_postfix"):
                current_lr = optimizer.param_groups[0]["lr"]
                iterator.set_postfix(loss=f"{running_loss:.4f}", lr=f"{current_lr:.2e}")

    elapsed = time.time() - start_time
    total_loss = reduce_sum(total_loss, device, world_size)
    total_count = int(reduce_sum(float(total_count), device, world_size))
    mean_loss = total_loss / max(total_count, 1)

    return {
        "loss": mean_loss,
        "samples": total_count,
        "seconds": elapsed,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    total_sq = 0.0
    total_bias = 0.0
    total_count = 0

    iterator = loader
    if is_main_process(rank):
        iterator = tqdm(
            loader,
            total=len(loader),
            desc="Validation",
            leave=True,
        )

    for batch in iterator:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        conf = batch["conf"].to(device, non_blocking=True)
        n_points = batch["n_points"].to(device, non_blocking=True)
        residual_std = batch["residual_std"].to(device, non_blocking=True)

        with nullcontext():
            pred = (
                model(
                    x,
                    conf=conf,
                    n_points=n_points,
                    residual_std=residual_std,
                    return_aux=False,
                )
                if USE_FUZZY_LOGIC
                else model(x)
            )
            loss = criterion(pred, y).mean()

        err = pred - y
        batch_size = x.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_abs += float(torch.sum(torch.abs(err)).item())
        total_sq += float(torch.sum(err * err).item())
        total_bias += float(torch.sum(err).item())
        total_count += batch_size

    total_loss = reduce_sum(total_loss, device, world_size)
    total_abs = reduce_sum(total_abs, device, world_size)
    total_sq = reduce_sum(total_sq, device, world_size)
    total_bias = reduce_sum(total_bias, device, world_size)
    total_count = int(reduce_sum(float(total_count), device, world_size))

    return {
        "loss": total_loss / max(total_count, 1),
        "mae": total_abs / max(total_count, 1),
        "rmse": math.sqrt(total_sq / max(total_count, 1)),
        "bias": total_bias / max(total_count, 1),
        "samples": total_count,
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_rmse: float,
) -> None:
    state = {
        "epoch": epoch,
        "model": model.module.state_dict() if is_parallel_model(model) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_rmse": best_val_rmse,
        "model_cfg": asdict(MODEL_CFG),
        "train_cfg": {
            "train_mode": TRAIN_MODE,
            "gpu_ids": GPU_IDS,
            "epochs": EPOCHS,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "plateau_factor": PLATEAU_FACTOR,
            "plateau_patience": PLATEAU_PATIENCE,
            "plateau_threshold": PLATEAU_THRESHOLD,
            "min_lr": MIN_LR,
            "loss_name": LOSS_NAME,
            "huber_delta": HUBER_DELTA,
            "use_fuzzy_logic": USE_FUZZY_LOGIC,
            "fuzzy_use_loss_weight": FUZZY_USE_LOSS_WEIGHT,
            "fuzzy_use_feature_gate": FUZZY_USE_FEATURE_GATE,
            "fuzzy_use_n_points": FUZZY_USE_N_POINTS,
            "fuzzy_use_residual_std": FUZZY_USE_RESIDUAL_STD,
            "fuzzy_weight_mode": FUZZY_WEIGHT_MODE,
            "use_amp": USE_AMP,
            "amp_dtype": AMP_DTYPE,
        },
    }
    torch.save(state, path)


def write_history_csv(history: list[dict[str, float]], csv_path: Path) -> None:
    if not history:
        return

    fieldnames = list(history[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def build_dataloader(
    dataset: Dataset,
    split: str,
    distributed: bool,
    world_size: int,
    rank: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    train_mode = TRAIN_MODE.lower()
    shuffle = split == "train"
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        shuffle = False

    if distributed:
        batch_size = BATCH_SIZE_PER_GPU
    elif train_mode == "dp":
        batch_size = BATCH_SIZE_PER_GPU * len(GPU_IDS)
    else:
        batch_size = BATCH_SIZE_PER_GPU

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    return loader, sampler


def main() -> None:
    distributed, rank, world_size, local_rank = setup_distributed()
    setup_seed(SEED + rank)

    train_mode = TRAIN_MODE.lower()
    available_gpu_count = torch.cuda.device_count()

    if train_mode == "ddp":
        if not distributed:
            raise RuntimeError(
                "TRAIN_MODE='ddp' requires torchrun. Example: "
                "'CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_mamba_swin.py'"
            )
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        effective_world_size = world_size
    elif train_mode == "dp":
        if not torch.cuda.is_available():
            raise RuntimeError("TRAIN_MODE='dp' requires CUDA, but no GPU is available.")
        if len(GPU_IDS) < 2:
            raise RuntimeError("TRAIN_MODE='dp' requires at least 2 GPU ids in GPU_IDS.")
        if max(GPU_IDS) >= available_gpu_count:
            raise RuntimeError(
                f"GPU_IDS={GPU_IDS} exceeds available GPU count ({available_gpu_count})."
            )
        torch.cuda.set_device(GPU_IDS[0])
        device = torch.device(f"cuda:{GPU_IDS[0]}")
        effective_world_size = len(GPU_IDS)
    elif train_mode == "single":
        if torch.cuda.is_available() and len(GPU_IDS) > 0:
            if GPU_IDS[0] >= available_gpu_count:
                raise RuntimeError(
                    f"GPU_IDS[0]={GPU_IDS[0]} exceeds available GPU count ({available_gpu_count})."
                )
            torch.cuda.set_device(GPU_IDS[0])
            device = torch.device(f"cuda:{GPU_IDS[0]}")
        else:
            device = torch.device("cpu")
        effective_world_size = 1
    else:
        raise ValueError("TRAIN_MODE must be one of: 'single', 'dp', 'ddp'.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUTPUT_DIR / CHECKPOINT_DIRNAME
    best_dir = OUTPUT_DIR / BEST_DIRNAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = best_dir / BEST_FILENAME

    train_dataset = PatchH5Dataset(TRAIN_PATCH_H5)
    val_dataset = PatchH5Dataset(VAL_PATCH_H5)

    train_loader, train_sampler = build_dataloader(train_dataset, "train", distributed, world_size, rank)
    val_loader, _ = build_dataloader(val_dataset, "val", distributed, world_size, rank)
    train_batch_size = train_loader.batch_size
    val_batch_size = val_loader.batch_size

    model = MambaSwinRegressor(MODEL_CFG).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif train_mode == "dp":
        model = nn.DataParallel(model, device_ids=GPU_IDS, output_device=GPU_IDS[0])

    criterion = create_loss().to(device)
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)

    scaler = None
    if USE_AMP and device.type == "cuda" and AMP_DTYPE.lower() == "fp16":
        scaler = torch.amp.GradScaler("cuda")

    if is_main_process(rank):
        print(f"Device: {device}")
        print(f"Train mode: {train_mode}")
        print(f"GPU ids: {GPU_IDS if device.type == 'cuda' else []}")
        print(f"Distributed: {distributed}, world_size={effective_world_size}")
        print(f"Available GPUs: {available_gpu_count}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples:   {len(val_dataset)}")
        print(f"Train steps/epoch: {len(train_loader)}")
        print(f"Val steps:         {len(val_loader)}")
        print(f"Batch size per GPU: {BATCH_SIZE_PER_GPU}")
        print(f"Train loader batch size: {train_batch_size}")
        print(f"Val loader batch size:   {val_batch_size}")
        print(f"Effective total batch size: {BATCH_SIZE_PER_GPU * effective_world_size}")
        print(f"Num workers: {NUM_WORKERS}")
        print(f"Preload patches to memory: {PRELOAD_PATCHES_TO_MEMORY}")
        print(f"Use AMP: {USE_AMP}")
        print(f"AMP dtype: {AMP_DTYPE}")
        print(f"GradScaler enabled: {scaler is not None}")
        print(f"Use fuzzy logic: {USE_FUZZY_LOGIC}")
        print(f"Fuzzy loss weight: {FUZZY_USE_LOSS_WEIGHT}")
        print(f"Fuzzy feature gate: {FUZZY_USE_FEATURE_GATE}")
        print(f"Fuzzy use n_points: {FUZZY_USE_N_POINTS}")
        print(f"Fuzzy use residual_std: {FUZZY_USE_RESIDUAL_STD}")
        print(f"Fuzzy weight mode: {FUZZY_WEIGHT_MODE}")

    history = []
    best_val_rmse = float("inf")

    if is_main_process(rank):
        config_dump = {
            "model_cfg": asdict(MODEL_CFG),
            "train_cfg": {
                "train_patch_h5": str(TRAIN_PATCH_H5),
                "val_patch_h5": str(VAL_PATCH_H5),
                "output_dir": str(OUTPUT_DIR),
                "checkpoint_dir": str(checkpoint_dir),
                "best_model_path": str(best_model_path),
                "train_mode": train_mode,
                "gpu_ids": GPU_IDS,
                "epochs": EPOCHS,
                "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
                "world_size": effective_world_size,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "plateau_factor": PLATEAU_FACTOR,
                "plateau_patience": PLATEAU_PATIENCE,
                "plateau_threshold": PLATEAU_THRESHOLD,
                "min_lr": MIN_LR,
                "loss_name": LOSS_NAME,
                "huber_delta": HUBER_DELTA,
                "use_fuzzy_logic": USE_FUZZY_LOGIC,
                "fuzzy_use_loss_weight": FUZZY_USE_LOSS_WEIGHT,
                "fuzzy_use_feature_gate": FUZZY_USE_FEATURE_GATE,
                "fuzzy_use_n_points": FUZZY_USE_N_POINTS,
                "fuzzy_use_residual_std": FUZZY_USE_RESIDUAL_STD,
                "fuzzy_weight_mode": FUZZY_WEIGHT_MODE,
                "use_amp": USE_AMP,
                "amp_dtype": AMP_DTYPE,
            },
        }
        (OUTPUT_DIR / "config.json").write_text(json.dumps(config_dump, indent=2), encoding="utf-8")

    for epoch in range(1, EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        scheduler.step(val_metrics["rmse"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_bias": val_metrics["bias"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        if is_main_process(rank):
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_mae={val_metrics['mae']:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} "
                f"val_bias={val_metrics['bias']:.6f}"
            )

            (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            write_history_csv(history, OUTPUT_DIR / "history.csv")

            if SAVE_EVERY_EPOCH:
                save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_rmse,
                )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                save_checkpoint(
                    best_model_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_rmse,
                )

        if distributed:
            dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
