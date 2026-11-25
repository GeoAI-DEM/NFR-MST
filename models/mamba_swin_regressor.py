#!/usr/bin/env python3
"""
Task-specific Mamba + Swin regressor for GEBCO DEM correction.

Design:
    - Input: patch tensor [B, C, H, W]
    - Swin stem: patch embedding with patch_size=1 and local BasicLayer blocks
    - Mamba: sequence modeling on flattened spatial tokens
    - Optional neuro-fuzzy confidence module:
        * explicit Low/Medium/High memberships from confidence
        * learned fuzzy rule activation
        * defuzzified sample reliability weight
        * optional feature gating before the regression head
    - Head: regress the center-cell residual
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
    """
    Import only mamba_simple from the local repo without triggering
    mamba_ssm/__init__.py, which eagerly imports newer Mamba2/Mamba3 code paths.
    Those newer paths require Triton APIs such as `triton.set_allocator`
    that are not needed for this project's MambaSimple backbone.
    """
    package_root = MAMBA_ROOT / "mamba_ssm"

    if "mamba_ssm" not in sys.modules:
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = [str(package_root)]  # mark as package
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
class MambaSwinRegressorConfig:
    patch_size: int = 33
    in_chans: int = 6
    embed_dim: int = 64
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
    fuzzy_enabled: bool = False
    fuzzy_use_loss_weight: bool = True
    fuzzy_use_feature_gate: bool = False
    fuzzy_use_n_points: bool = True
    fuzzy_use_residual_std: bool = True
    fuzzy_hidden_dim: int = 16
    fuzzy_membership_low: tuple[float, float, float, float] = (0.60, 0.60, 0.68, 0.77)
    fuzzy_membership_medium: tuple[float, float, float] = (0.68, 0.77, 0.85)
    fuzzy_membership_high: tuple[float, float, float, float] = (0.85, 0.91, 1.00, 1.00)
    fuzzy_n_points_low: tuple[float, float, float, float] = (3.0, 3.0, 6.0, 15.0)
    fuzzy_n_points_medium: tuple[float, float, float] = (8.0, 20.0, 60.0)
    fuzzy_n_points_high: tuple[float, float, float, float] = (30.0, 100.0, 1000.0, 1000.0)
    fuzzy_residual_std_low: tuple[float, float, float, float] = (0.0, 0.0, 0.15, 0.35)
    fuzzy_residual_std_medium: tuple[float, float, float] = (0.20, 0.45, 0.90)
    fuzzy_residual_std_high: tuple[float, float, float, float] = (0.70, 1.20, 5.0, 5.0)
    fuzzy_output_centers: tuple[float, float, float] = (0.25, 0.60, 1.00)


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


class FuzzyConfidenceNet(nn.Module):
    def __init__(self, cfg: MambaSwinRegressorConfig, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.use_loss_weight = cfg.fuzzy_use_loss_weight
        self.use_feature_gate = cfg.fuzzy_use_feature_gate
        self.use_n_points = cfg.fuzzy_use_n_points
        self.use_residual_std = cfg.fuzzy_use_residual_std
        self.register_buffer("low_params", torch.tensor(cfg.fuzzy_membership_low, dtype=torch.float32))
        self.register_buffer("medium_params", torch.tensor(cfg.fuzzy_membership_medium, dtype=torch.float32))
        self.register_buffer("high_params", torch.tensor(cfg.fuzzy_membership_high, dtype=torch.float32))
        self.register_buffer("n_points_low_params", torch.tensor(cfg.fuzzy_n_points_low, dtype=torch.float32))
        self.register_buffer("n_points_medium_params", torch.tensor(cfg.fuzzy_n_points_medium, dtype=torch.float32))
        self.register_buffer("n_points_high_params", torch.tensor(cfg.fuzzy_n_points_high, dtype=torch.float32))
        self.register_buffer("residual_std_low_params", torch.tensor(cfg.fuzzy_residual_std_low, dtype=torch.float32))
        self.register_buffer("residual_std_medium_params", torch.tensor(cfg.fuzzy_residual_std_medium, dtype=torch.float32))
        self.register_buffer("residual_std_high_params", torch.tensor(cfg.fuzzy_residual_std_high, dtype=torch.float32))
        self.register_buffer("output_centers", torch.tensor(cfg.fuzzy_output_centers, dtype=torch.float32))

        raw_feature_dim = 1
        membership_dim = 3
        if self.use_n_points:
            raw_feature_dim += 1
            membership_dim += 3
        if self.use_residual_std:
            raw_feature_dim += 1
            membership_dim += 3

        self.backbone = nn.Sequential(
            nn.Linear(raw_feature_dim + membership_dim, cfg.fuzzy_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.fuzzy_hidden_dim, cfg.fuzzy_hidden_dim),
            nn.GELU(),
        )
        self.rule_head = nn.Linear(cfg.fuzzy_hidden_dim, 3) if self.use_loss_weight else None
        self.gate_head = nn.Linear(cfg.fuzzy_hidden_dim, feature_dim) if self.use_feature_gate else None

    @staticmethod
    def _trapmf(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        a, b, c, d = params.unbind()
        eps = 1.0e-8
        y = torch.zeros_like(x)
        if float(b.item()) == float(a.item()):
            y = torch.where(x <= b, torch.ones_like(x), y)
        else:
            left_mask = (x > a) & (x < b)
            y = torch.where(left_mask, (x - a) / (b - a + eps), y)
        plateau_mask = (x >= b) & (x <= c)
        y = torch.where(plateau_mask, torch.ones_like(y), y)
        if float(d.item()) == float(c.item()):
            y = torch.where(x >= c, torch.ones_like(x), y)
        else:
            right_mask = (x > c) & (x < d)
            y = torch.where(right_mask, (d - x) / (d - c + eps), y)
        return torch.clamp(y, 0.0, 1.0)

    @staticmethod
    def _trimf(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        a, b, c = params.unbind()
        eps = 1.0e-8
        left = torch.clamp((x - a) / (b - a + eps), 0.0, 1.0)
        right = torch.clamp((c - x) / (c - b + eps), 0.0, 1.0)
        return torch.clamp(torch.minimum(left, right), 0.0, 1.0)

    def fuzzify_conf(self, conf: torch.Tensor) -> torch.Tensor:
        conf = conf.view(-1, 1)
        mu_low = self._trapmf(conf, self.low_params).view(-1, 1)
        mu_med = self._trimf(conf, self.medium_params).view(-1, 1)
        mu_high = self._trapmf(conf, self.high_params).view(-1, 1)
        return torch.cat([mu_low, mu_med, mu_high], dim=-1)

    def fuzzify_n_points(self, n_points: torch.Tensor) -> torch.Tensor:
        n_points = n_points.view(-1, 1)
        mu_low = self._trapmf(n_points, self.n_points_low_params).view(-1, 1)
        mu_med = self._trimf(n_points, self.n_points_medium_params).view(-1, 1)
        mu_high = self._trapmf(n_points, self.n_points_high_params).view(-1, 1)
        return torch.cat([mu_low, mu_med, mu_high], dim=-1)

    def fuzzify_residual_std(self, residual_std: torch.Tensor) -> torch.Tensor:
        residual_std = residual_std.view(-1, 1)
        mu_low = self._trapmf(residual_std, self.residual_std_low_params).view(-1, 1)
        mu_med = self._trimf(residual_std, self.residual_std_medium_params).view(-1, 1)
        mu_high = self._trapmf(residual_std, self.residual_std_high_params).view(-1, 1)
        return torch.cat([mu_low, mu_med, mu_high], dim=-1)

    def forward(
        self,
        conf: torch.Tensor,
        n_points: torch.Tensor | None = None,
        residual_std: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        conf = conf.view(-1, 1)
        conf_memberships = self.fuzzify_conf(conf)

        raw_features = [conf]
        memberships = [conf_memberships]

        n_points_memberships = None
        if self.use_n_points:
            if n_points is None:
                raise ValueError("`n_points` must be provided when fuzzy_use_n_points=True.")
            n_points = n_points.view(-1, 1).to(conf.dtype)
            raw_features.append(torch.log1p(n_points))
            n_points_memberships = self.fuzzify_n_points(n_points)
            memberships.append(n_points_memberships)

        residual_std_memberships = None
        if self.use_residual_std:
            if residual_std is None:
                raise ValueError("`residual_std` must be provided when fuzzy_use_residual_std=True.")
            residual_std = residual_std.view(-1, 1).to(conf.dtype)
            raw_features.append(residual_std)
            residual_std_memberships = self.fuzzify_residual_std(residual_std)
            memberships.append(residual_std_memberships)

        membership_tensor = torch.cat(memberships, dim=-1)
        fuzzy_input = torch.cat(raw_features + [membership_tensor], dim=-1)
        hidden = self.backbone(fuzzy_input)

        rule_scores = None
        fuzzy_weight = None
        if self.rule_head is not None:
            rule_scores = torch.softmax(self.rule_head(hidden), dim=-1)
            fuzzy_weight = torch.sum(rule_scores * self.output_centers.unsqueeze(0), dim=-1)

        gate = None
        if self.gate_head is not None:
            gate = torch.sigmoid(self.gate_head(hidden))

        return {
            "conf_memberships": conf_memberships,
            "n_points_memberships": n_points_memberships,
            "residual_std_memberships": residual_std_memberships,
            "memberships": membership_tensor,
            "rule_scores": rule_scores,
            "fuzzy_weight": fuzzy_weight,
            "feature_gate": gate,
        }


class MambaSwinRegressor(nn.Module):
    def __init__(self, cfg: MambaSwinRegressorConfig) -> None:
        super().__init__()
        if Mamba is None:
            raise RuntimeError(
                "Failed to import local Mamba implementation from "
                f"{MAMBA_ROOT}. Original error: {_MAMBA_IMPORT_ERROR}"
            )

        if cfg.patch_size % cfg.window_size != 0:
            raise ValueError(
                f"patch_size ({cfg.patch_size}) must be divisible by window_size ({cfg.window_size})."
            )
        if len(cfg.swin_depths) != len(cfg.swin_num_heads):
            raise ValueError("swin_depths and swin_num_heads must have the same length.")

        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim
        self.token_grid_size = cfg.patch_size
        self.num_tokens = self.token_grid_size * self.token_grid_size
        self.center_token_index = (self.token_grid_size // 2) * self.token_grid_size + (self.token_grid_size // 2)

        self.patch_embed = PatchEmbed(
            img_size=cfg.patch_size,
            patch_size=1,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            norm_layer=nn.LayerNorm,
        )
        self.pos_drop = nn.Dropout(p=cfg.drop_rate)

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
        self.head_input_dim = cfg.embed_dim * 2
        self.fuzzy_net = FuzzyConfidenceNet(cfg, self.head_input_dim) if cfg.fuzzy_enabled else None
        self.head = ResidualRegressorHead(self.head_input_dim, cfg.head_hidden_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # [B, L, C]
        x = self.pos_drop(x)

        for layer in self.swin_layers:
            x = layer(x)

        x = self.pre_mamba_norm(x)
        for layer in self.mamba_layers:
            x = x + layer(x)
        x = self.post_mamba_norm(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        conf: torch.Tensor | None = None,
        n_points: torch.Tensor | None = None,
        residual_std: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            pred_residual: [B]
        """
        tokens = self.forward_features(x)
        center_token = tokens[:, self.center_token_index, :]
        mean_token = tokens.mean(dim=1)
        fused_feature = torch.cat([center_token, mean_token], dim=-1)

        fuzzy_outputs: dict[str, torch.Tensor | None] = {}
        if self.fuzzy_net is not None:
            if conf is None:
                raise ValueError("Confidence input `conf` must be provided when fuzzy module is enabled.")
            fuzzy_outputs = self.fuzzy_net(conf, n_points=n_points, residual_std=residual_std)
            if self.cfg.fuzzy_use_feature_gate:
                gate = fuzzy_outputs["feature_gate"]
                if gate is None:
                    raise RuntimeError("Feature gate is missing although fuzzy_use_feature_gate=True.")
                fused_feature = fused_feature * gate

        pred = self.head(fused_feature)
        if return_aux:
            return {
                "pred": pred,
                "fused_feature": fused_feature,
                **fuzzy_outputs,
            }
        return pred


def smoke_test(device: str = "cpu") -> None:
    cfg = MambaSwinRegressorConfig()
    model = MambaSwinRegressor(cfg).to(device)
    x = torch.randn(2, cfg.in_chans, cfg.patch_size, cfg.patch_size, device=device)
    y = model(x)
    print("smoke_test_ok")
    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(y.shape))


if __name__ == "__main__":
    smoke_test()
