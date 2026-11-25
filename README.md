# NFR-MST

Neuro-fuzzy reliability-guided Mamba-Swin Transformer for coastal GEBCO DEM
correction using ICESat-2 ATL24. The model estimates the center-cell residual
`GEBCO - ATL24` from six-channel terrain patches. Subtract the predicted residual
from the original GEBCO elevation to obtain the corrected elevation.

## Data availability

The processed datasets used in this study will be made publicly available upon publication of the associated paper.

## Setup

The model was developed with Python 3.10, PyTorch 2.5.1, CUDA 12.1 and timm 0.9.16.
A compatible NVIDIA GPU and CUDA toolkit (`nvcc`) are needed to build and run the
selective-scan extension. From the repository root:

```sh
python -m pip install setuptools wheel packaging ninja
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
MAMBA_FORCE_BUILD=TRUE python -m pip install ./mamba-main --no-build-isolation --no-deps
```

The local model imports Mamba v1 directly. The full upstream Mamba package also
declares dependencies for newer backbones that this project does not use.

## Training and evaluation

Provide your own `train_patches.h5`, `val_patches.h5`, and `test_patches.h5`.
The HDF5 field `x` contains `[N, 6, 33, 33]` patches in elevation, slope,
curvature, roughness, latitude, and longitude channel order. Each record also has
`y` (target residual), `conf`, `n_points`, `residual_std`, `row`, `col`,
and `val_block_id`. The training loader additionally expects `weight`.
Input paths and training settings are grouped near the top of each entry point.

```sh
torchrun --nproc_per_node=2 train_mamba_swin.py
python evaluate_test_mamba_swin.py
```

The training entry point preserves the current gate-only defaults. For full
reliability guidance, set both `FUZZY_USE_LOSS_WEIGHT` and
`FUZZY_USE_FEATURE_GATE` to `True` and select a separate output directory.
Update `CHECKPOINT_PATH` in the evaluator when evaluating a different run.
The evaluator reports overall and regional metrics in JSON and CSV.

The vendored Mamba and Swin Transformer sources retain their upstream notices
and licenses in their respective directories.
