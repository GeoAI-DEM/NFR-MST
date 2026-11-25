#!/usr/bin/env python3
"""
Evaluate a trained Mamba+Swin regressor on the 8 selected test regions.

Input:
    - test_patches.h5
    - best checkpoint .pt

Output:
    - test_metrics.json
    - test_block_metrics.csv

All paths and options are defined in CONFIG below.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models import MambaSwinRegressor, MambaSwinRegressorConfig


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "run_fuzzy_gate_only" / "best" / "best.pt"
TEST_PATCH_H5 = PROJECT_ROOT / "test_patches.h5"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_eval"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024
NUM_WORKERS = 0
PIN_MEMORY = True
PRELOAD_TO_MEMORY = True
PATCH_CENTER = 16

BLOCK_NAMES = {
    0: "asia_beibu_gulf_east",
    1: "asia_zhoushan_south",
    2: "europe_mediterranean_greece",
    4: "africa_zanzibar",
    5: "north_america_florida_keys",
    6: "south_america_brazil_ne",
    7: "oceania_torres_strait",
    8: "caribbean_bahamas",
}

# Existing H5 files keep the original source block ids. Results are reported
# with compact display ids 1-8 after excluding the former block_id=3.
DISPLAY_BLOCK_IDS = {
    0: 1,
    1: 2,
    2: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
}


class TestPatchDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: Path, preload: bool = True) -> None:
        self.h5_path = h5_path
        self.preload = preload
        self._file = None

        self._x = None
        self._y = None
        self._row = None
        self._col = None
        self._conf = None
        self._n_points = None
        self._residual_std = None
        self._val_block_id = None

        with h5py.File(h5_path, "r") as f:
            self.length = int(f["x"].shape[0])
            if preload:
                self._x = f["x"][:].astype(np.float32, copy=False)
                self._y = f["y"][:].astype(np.float32, copy=False)
                self._row = f["row"][:].astype(np.int64, copy=False)
                self._col = f["col"][:].astype(np.int64, copy=False)
                self._conf = f["conf"][:].astype(np.float32, copy=False)
                self._n_points = f["n_points"][:].astype(np.int64, copy=False)
                self._residual_std = f["residual_std"][:].astype(np.float32, copy=False)
                self._val_block_id = f["val_block_id"][:].astype(np.int64, copy=False)

    def __len__(self) -> int:
        return self.length

    def _ensure_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.preload:
            x = self._x[idx]
            y = self._y[idx]
            row = self._row[idx]
            col = self._col[idx]
            conf = self._conf[idx]
            n_points = self._n_points[idx]
            residual_std = self._residual_std[idx]
            block_id = self._val_block_id[idx]
        else:
            f = self._ensure_file()
            x = f["x"][idx].astype(np.float32)
            y = np.float32(f["y"][idx])
            row = np.int64(f["row"][idx])
            col = np.int64(f["col"][idx])
            conf = np.float32(f["conf"][idx])
            n_points = np.int64(f["n_points"][idx])
            residual_std = np.float32(f["residual_std"][idx])
            block_id = np.int64(f["val_block_id"][idx])

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "row": torch.tensor(row, dtype=torch.int64),
            "col": torch.tensor(col, dtype=torch.int64),
            "conf": torch.tensor(conf, dtype=torch.float32),
            "n_points": torch.tensor(n_points, dtype=torch.int64),
            "residual_std": torch.tensor(residual_std, dtype=torch.float32),
            "val_block_id": torch.tensor(block_id, dtype=torch.int64),
        }


def build_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict, MambaSwinRegressorConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint.get("model_cfg")
    if model_cfg is None:
        raise RuntimeError(f"'model_cfg' not found in checkpoint: {checkpoint_path}")

    cfg = MambaSwinRegressorConfig(**model_cfg)
    model = MambaSwinRegressor(cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint, cfg


def make_loader(dataset: TestPatchDataset) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=False,
    )


def init_metric_state() -> dict[str, float]:
    return {
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "sum_bias": 0.0,
        "count": 0,
    }


def update_metric_state(state: dict[str, float], err: torch.Tensor | np.ndarray) -> None:
    if isinstance(err, np.ndarray):
        state["sum_abs"] += float(np.abs(err).sum())
        state["sum_sq"] += float(np.square(err).sum())
        state["sum_bias"] += float(err.sum())
        state["count"] += int(err.size)
        return
    state["sum_abs"] += float(torch.sum(torch.abs(err)).item())
    state["sum_sq"] += float(torch.sum(err * err).item())
    state["sum_bias"] += float(torch.sum(err).item())
    state["count"] += int(err.numel())


def finalize_metric_state(state: dict[str, float]) -> dict[str, float]:
    count = max(int(state["count"]), 1)
    return {
        "count": int(state["count"]),
        "mae": state["sum_abs"] / count,
        "rmse": math.sqrt(state["sum_sq"] / count),
        "bias": state["sum_bias"] / count,
    }



@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_fuzzy_logic: bool,
) -> tuple[
    dict[str, float],
    dict[str, float],
    list[dict[str, float]],
]:
    overall_corrected = init_metric_state()
    overall_raw = init_metric_state()
    block_states_corrected = {block_id: init_metric_state() for block_id in BLOCK_NAMES}
    block_states_raw = {block_id: init_metric_state() for block_id in BLOCK_NAMES}

    iterator = tqdm(loader, total=len(loader), desc="Testing", leave=True)
    for batch in iterator:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        conf = batch["conf"].to(device, non_blocking=True)
        n_points = batch["n_points"].to(device, non_blocking=True)
        residual_std = batch["residual_std"].to(device, non_blocking=True)
        block_id = batch["val_block_id"].to(device, non_blocking=True)

        pred = (
            model(
                x,
                conf=conf,
                n_points=n_points,
                residual_std=residual_std,
                return_aux=False,
            )
            if use_fuzzy_logic
            else model(x)
        )

        eval_mask = torch.zeros_like(block_id, dtype=torch.bool)
        for evaluated_block_id in BLOCK_NAMES:
            eval_mask |= block_id == evaluated_block_id
        if not torch.any(eval_mask):
            continue

        y = y[eval_mask]
        pred = pred[eval_mask]
        block_id = block_id[eval_mask]
        raw_err = y
        corrected_err = y - pred

        update_metric_state(overall_raw, raw_err)
        update_metric_state(overall_corrected, corrected_err)

        for current_block_id in torch.unique(block_id).tolist():
            mask = block_id == current_block_id
            update_metric_state(block_states_raw[int(current_block_id)], raw_err[mask])
            update_metric_state(block_states_corrected[int(current_block_id)], corrected_err[mask])

    overall_raw_metrics = finalize_metric_state(overall_raw)
    overall_corrected_metrics = finalize_metric_state(overall_corrected)
    block_metrics = []
    for block_id, block_name in BLOCK_NAMES.items():
        raw_metrics = finalize_metric_state(block_states_raw[block_id])
        corrected_metrics = finalize_metric_state(block_states_corrected[block_id])
        mae_improve_pct = (
            100.0 * (raw_metrics["mae"] - corrected_metrics["mae"]) / raw_metrics["mae"]
            if raw_metrics["mae"] > 0
            else float("nan")
        )
        rmse_improve_pct = (
            100.0 * (raw_metrics["rmse"] - corrected_metrics["rmse"]) / raw_metrics["rmse"]
            if raw_metrics["rmse"] > 0
            else float("nan")
        )
        block_metrics.append(
            {
                "block_id": DISPLAY_BLOCK_IDS[block_id],
                "block_name": block_name,
                "count": corrected_metrics["count"],
                "raw_mae": raw_metrics["mae"],
                "raw_rmse": raw_metrics["rmse"],
                "raw_bias": raw_metrics["bias"],
                "corrected_mae": corrected_metrics["mae"],
                "corrected_rmse": corrected_metrics["rmse"],
                "corrected_bias": corrected_metrics["bias"],
                "mae_improve_pct": mae_improve_pct,
                "rmse_improve_pct": rmse_improve_pct,
            }
        )

    return (
        overall_raw_metrics,
        overall_corrected_metrics,
        block_metrics,
    )


def write_block_metrics_csv(path: Path, block_metrics: list[dict[str, float]]) -> None:
    fieldnames = [
        "block_id",
        "block_name",
        "count",
        "raw_mae",
        "raw_rmse",
        "raw_bias",
        "corrected_mae",
        "corrected_rmse",
        "corrected_bias",
        "mae_improve_pct",
        "rmse_improve_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in block_metrics:
            writer.writerow(row)


def compute_macro_average(block_metrics: list[dict[str, float]], prefix: str) -> dict[str, float]:
    valid = [row for row in block_metrics if row["count"] > 0]
    if not valid:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    return {
        "mae": float(np.mean([row[f"{prefix}_mae"] for row in valid])),
        "rmse": float(np.mean([row[f"{prefix}_rmse"] for row in valid])),
        "bias": float(np.mean([row[f"{prefix}_bias"] for row in valid])),
    }



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    dataset = TestPatchDataset(TEST_PATCH_H5, preload=PRELOAD_TO_MEMORY)
    loader = make_loader(dataset)
    model, checkpoint, model_cfg = build_model(CHECKPOINT_PATH, device)

    (
        overall_raw_metrics,
        overall_corrected_metrics,
        block_metrics,
    ) = evaluate(
        model,
        loader,
        device,
        use_fuzzy_logic=bool(model_cfg.fuzzy_enabled),
    )
    macro_raw = compute_macro_average(block_metrics, prefix="raw")
    macro_corrected = compute_macro_average(block_metrics, prefix="corrected")
    overall_mae_improve_pct = (
        100.0 * (overall_raw_metrics["mae"] - overall_corrected_metrics["mae"]) / overall_raw_metrics["mae"]
        if overall_raw_metrics["mae"] > 0
        else float("nan")
    )
    overall_rmse_improve_pct = (
        100.0 * (overall_raw_metrics["rmse"] - overall_corrected_metrics["rmse"]) / overall_raw_metrics["rmse"]
        if overall_raw_metrics["rmse"] > 0
        else float("nan")
    )
    macro_mae_improve_pct = (
        100.0 * (macro_raw["mae"] - macro_corrected["mae"]) / macro_raw["mae"]
        if macro_raw["mae"] > 0
        else float("nan")
    )
    macro_rmse_improve_pct = (
        100.0 * (macro_raw["rmse"] - macro_corrected["rmse"]) / macro_raw["rmse"]
        if macro_raw["rmse"] > 0
        else float("nan")
    )
    result = {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "test_patch_h5": str(TEST_PATCH_H5),
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "preload_to_memory": PRELOAD_TO_MEMORY,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_rmse": checkpoint.get("best_val_rmse"),
        "fuzzy_enabled": bool(model_cfg.fuzzy_enabled),
        "fuzzy_use_feature_gate": bool(model_cfg.fuzzy_use_feature_gate),
        "fuzzy_use_n_points": bool(model_cfg.fuzzy_use_n_points),
        "fuzzy_use_residual_std": bool(model_cfg.fuzzy_use_residual_std),
        "overall_raw_gebco": overall_raw_metrics,
        "overall_corrected": overall_corrected_metrics,
        "overall_improvement_pct": {
            "mae": overall_mae_improve_pct,
            "rmse": overall_rmse_improve_pct,
        },
        "macro_avg_over_8_blocks_raw": macro_raw,
        "macro_avg_over_8_blocks_corrected": macro_corrected,
        "macro_avg_improvement_pct": {
            "mae": macro_mae_improve_pct,
            "rmse": macro_rmse_improve_pct,
        },
        "blocks": block_metrics,
    }

    json_path = OUTPUT_DIR / "test_metrics.json"
    csv_path = OUTPUT_DIR / "test_block_metrics.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_block_metrics_csv(csv_path, block_metrics)

    print("Finished test evaluation.")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Test patches: {TEST_PATCH_H5}")
    print(f"Output JSON: {json_path}")
    print(f"Output CSV: {csv_path}")
    print("")
    print(
        f"Overall Raw GEBCO: count={overall_raw_metrics['count']} "
        f"MAE={overall_raw_metrics['mae']:.6f} "
        f"RMSE={overall_raw_metrics['rmse']:.6f} "
        f"Bias={overall_raw_metrics['bias']:.6f}"
    )
    print(
        f"Overall Corrected: count={overall_corrected_metrics['count']} "
        f"MAE={overall_corrected_metrics['mae']:.6f} "
        f"RMSE={overall_corrected_metrics['rmse']:.6f} "
        f"Bias={overall_corrected_metrics['bias']:.6f}"
    )
    print("")
    print(
        f"Overall Improvement: "
        f"MAE={overall_mae_improve_pct:.2f}% "
        f"RMSE={overall_rmse_improve_pct:.2f}%"
    )
    print(
        f"Macro Avg Raw over 8 blocks: "
        f"MAE={macro_raw['mae']:.6f} "
        f"RMSE={macro_raw['rmse']:.6f} "
        f"Bias={macro_raw['bias']:.6f}"
    )
    print(
        f"Macro Avg Corrected over 8 blocks: "
        f"MAE={macro_corrected['mae']:.6f} "
        f"RMSE={macro_corrected['rmse']:.6f} "
        f"Bias={macro_corrected['bias']:.6f}"
    )
    print(
        f"Macro Avg Improvement: "
        f"MAE={macro_mae_improve_pct:.2f}% "
        f"RMSE={macro_rmse_improve_pct:.2f}%"
    )
    print("")
    print("")
    print("Per-block metrics:")
    for row in block_metrics:
        print(
            f"  {row['block_id']}: {row['block_name']} | "
            f"count={row['count']} "
            f"raw_RMSE={row['raw_rmse']:.6f} "
            f"corrected_RMSE={row['corrected_rmse']:.6f} "
            f"RMSE_improve={row['rmse_improve_pct']:.2f}% "
            f"raw_MAE={row['raw_mae']:.6f} "
            f"corrected_MAE={row['corrected_mae']:.6f} "
            f"raw_bias={row['raw_bias']:.6f} "
            f"corrected_bias={row['corrected_bias']:.6f}"
        )


if __name__ == "__main__":
    main()
