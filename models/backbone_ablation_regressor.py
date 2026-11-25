#!/usr/bin/env python3
"""
Backbone ablation regressors for GEBCO DEM correction.

Supported modes:
    - cnn_only
    - mamba_only
    - swin_only
    - hybrid
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWIN_MODELS_DIR = PROJECT_ROOT / "Swin-Transformer-main" / "models"
MAMBA_ROOT = PROJECT_ROOT / "mamba-main"

if str(SWIN_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(SWIN_MODELS_DIR))
if str(MAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(MAMBA_ROOT))

from swin_transformer import BasicLayer, PatchEmbed  # type: ignore  # noqa: E402


def _import_local_mamba_simple():
    package_root = MAMBA_ROOT / "mamba_ssm"

    if "mamba_ssm" not in sys.modules:
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = [str(package_root)]
        pkg.__file__ = str(package_root / "__init__.py")
        sys.modules["mamba_ssm"] = pkg

    module = importlib.import_module("mamba_ssm.modules.mamba_simple")
    return module.Mamba


try:
    Mamba = _import_local_mamba_simple()  # type: ignore
except Exception as exc:  # pragma: no cover
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


@dataclass
class BackboneAblationRegressorConfig:
    patch_size: int = 33
    in_chans: int = 6
    embed_dim: int = 64
    backbone_mode: str = "hybrid"  # cnn_only, mamba_only, swin_only, hybrid
    cnn_depth: int = 8
    swin_depths: tuple[int, ...] = (4, 4)
    swin_num_heads: tuple[int, ...] = (4, 4)
    window_size: int = 11
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.05
    mamba_layers: int = 4
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    head_hidden_dim: int = 128
    norm_layer_eps: float = 1.0e-6


def _group_count(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResidualRegressorHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class BackboneAblationRegressor(nn.Module):
    def __init__(self, cfg: BackboneAblationRegressorConfig) -> None:
        super().__init__()
        mode = cfg.backbone_mode.lower()
        if mode not in {"cnn_only", "mamba_only", "swin_only", "hybrid"}:
            raise ValueError(f"Unsupported backbone_mode: {cfg.backbone_mode}")
        if cfg.patch_size % cfg.window_size != 0:
            raise ValueError(
                f"patch_size ({cfg.patch_size}) must be divisible by window_size ({cfg.window_size})."
            )
        if len(cfg.swin_depths) != len(cfg.swin_num_heads):
            raise ValueError("swin_depths and swin_num_heads must have the same length.")
        if mode in {"mamba_only", "hybrid"} and Mamba is None:
            raise RuntimeError(
                "Failed to import local Mamba implementation from "
                f"{MAMBA_ROOT}. Original error: {_MAMBA_IMPORT_ERROR}"
            )

        self.cfg = cfg
        self.mode = mode
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim
        self.token_grid_size = cfg.patch_size
        self.num_tokens = self.token_grid_size * self.token_grid_size
        self.center_token_index = (self.token_grid_size // 2) * self.token_grid_size + (self.token_grid_size // 2)

        self.use_cnn = mode == "cnn_only"
        self.use_swin = mode in {"swin_only", "hybrid"}
        self.use_mamba = mode in {"mamba_only", "hybrid"}

        if self.use_cnn:
            groups = _group_count(cfg.embed_dim)
            self.cnn_stem = nn.Sequential(
                nn.Conv2d(cfg.in_chans, cfg.embed_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(groups, cfg.embed_dim),
                nn.GELU(),
            )
            self.cnn_blocks = nn.ModuleList([ConvResidualBlock(cfg.embed_dim) for _ in range(cfg.cnn_depth)])
            self.cnn_token_norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_layer_eps)
        else:
            self.patch_embed = PatchEmbed(
                img_size=cfg.patch_size,
                patch_size=1,
                in_chans=cfg.in_chans,
                embed_dim=cfg.embed_dim,
                norm_layer=nn.LayerNorm,
            )
            self.pos_drop = nn.Dropout(p=cfg.drop_rate)

            if self.use_swin:
                dpr = torch.linspace(0, cfg.drop_path_rate, sum(cfg.swin_depths)).tolist()
                self.swin_layers = nn.ModuleList()
                dpr_ptr = 0
                for depth, num_heads in zip(cfg.swin_depths, cfg.swin_num_heads):
                    layer = BasicLayer(
                        dim=cfg.embed_dim,
                        input_resolution=(self.token_grid_size, self.token_grid_size),
                        depth=depth,
                        num_heads=num_heads,
                        window_size=cfg.window_size,
                        mlp_ratio=cfg.mlp_ratio,
                        qkv_bias=True,
                        qk_scale=None,
                        drop=cfg.drop_rate,
                        attn_drop=cfg.attn_drop_rate,
                        drop_path=dpr[dpr_ptr:dpr_ptr + depth],
                        norm_layer=nn.LayerNorm,
                        downsample=None,
                        use_checkpoint=False,
                        fused_window_process=False,
                    )
                    self.swin_layers.append(layer)
                    dpr_ptr += depth
            else:
                self.swin_layers = nn.ModuleList()

            if self.use_mamba:
                self.pre_mamba_norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_layer_eps)
                self.mamba_layers = nn.ModuleList(
                    [
                        Mamba(
                            d_model=cfg.embed_dim,
                            d_state=cfg.mamba_d_state,
                            d_conv=cfg.mamba_d_conv,
                            expand=cfg.mamba_expand,
                            use_fast_path=True,
                        )
                        for _ in range(cfg.mamba_layers)
                    ]
                )
                self.post_mamba_norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_layer_eps)
            else:
                self.mamba_layers = nn.ModuleList()

        self.head = ResidualRegressorHead(cfg.embed_dim * 2, cfg.head_hidden_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            x = self.cnn_stem(x)
            for block in self.cnn_blocks:
                x = block(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.cnn_token_norm(x)
            return x

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.swin_layers:
            x = layer(x)

        if self.use_mamba:
            x = self.pre_mamba_norm(x)
            for layer in self.mamba_layers:
                x = x + layer(x)
            x = self.post_mamba_norm(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_features(x)
        center_token = tokens[:, self.center_token_index, :]
        mean_token = tokens.mean(dim=1)
        fused = torch.cat([center_token, mean_token], dim=-1)
        return self.head(fused)


def smoke_test(device: str = "cpu") -> None:
    for mode in ("cnn_only", "mamba_only", "swin_only", "hybrid"):
        cfg = BackboneAblationRegressorConfig(backbone_mode=mode)
        model = BackboneAblationRegressor(cfg).to(device)
        x = torch.randn(2, cfg.in_chans, cfg.patch_size, cfg.patch_size, device=device)
        y = model(x)
        print(mode, tuple(y.shape))


if __name__ == "__main__":
    smoke_test()
