# Training loop for the FT-Transformer multi-horizon return model.
#
# Fits FTTransformerModel on the train split, validates each epoch, and saves the
# best checkpoint (by val loss) for evaluate.py / inference.py to load.
#
#   - AdamW (lr=1e-4, weight_decay=1e-5) + CosineAnnealingLR
#   - Loss: MSE, mean over batch, SUMMED over all len(HORIZONS) heads
#   - Modality dropout is applied automatically by the model while in train() mode
#   - Early stopping on val loss (patience=10)
#   - Best checkpoint -> backend/app/ml/checkpoints/best_model.pt
#   - Per epoch: train_loss, val_loss, direction_accuracy (sign match on 3M head)
#
# Entry point: run from the backend/ directory (there is no backend/__init__.py):
#     python -m app.ml.train
#
# Depends on: dataset.py, labels.py, backbone.py (FTTransformerModel), features.py

from __future__ import annotations

import logging
from pathlib import Path

import torch
from sqlalchemy.orm import Session
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .backbone import FTTransformerModel
from .dataset import DatasetBundle, build_datasets
from .features import HORIZONS

logger = logging.getLogger(__name__)

# ── Hyperparameters ──────────────────────────────────────────────────────────

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42

# The model is built with library defaults; the exact config is saved alongside
# the weights so evaluate.py rebuilds an identical architecture before loading.
MODEL_CONFIG: dict[str, float | int] = {
    "d_model": 192,
    "n_heads": 8,
    "num_layers": 3,
    "ffn_dim": 512,
    "dropout": 0.1,
    "modality_dropout_p": 0.3,
}

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best_model.pt"

# 3M horizon = 63 trading days; its column index drives direction accuracy.
_H3M = HORIZONS.index(63)


# ── Loss & metrics ───────────────────────────────────────────────────────────

def multi_horizon_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE averaged over the batch and **summed** over the horizon heads.

    This matches the "summed over all horizon heads" contract: plain
    ``nn.MSELoss()`` would also average over horizons, dividing the gradient by
    ``len(HORIZONS)`` and rescaling the effective learning rate.
    """
    return ((preds - targets) ** 2).sum(dim=1).mean()


def direction_accuracy(preds: torch.Tensor, targets: torch.Tensor, horizon_idx: int) -> float:
    """Fraction of rows where predicted and actual returns share a sign.

    Uses ``> 0`` rather than ``sign`` so a flat 0.0 prediction is treated as
    "not up" instead of matching only exact-zero targets.
    """
    if preds.shape[0] == 0:
        return float("nan")
    pred_up = preds[:, horizon_idx] > 0
    actual_up = targets[:, horizon_idx] > 0
    return (pred_up == actual_up).float().mean().item()


# ── Train / validate steps ───────────────────────────────────────────────────

def train_one_epoch(
    model: FTTransformerModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """Run one training epoch; returns the mean per-batch train loss."""
    model.train()  # enables random modality dropout inside the model
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = multi_horizon_mse(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_split(
    model: FTTransformerModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Return ``(loss, direction_accuracy_3M)`` over a loader (no dropout)."""
    model.eval()
    total_loss = 0.0
    n_rows = 0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        # Sum losses weighted by batch size so the mean is exact regardless of a
        # ragged final batch.
        total_loss += multi_horizon_mse(preds, y).item() * x.shape[0]
        correct += ((preds[:, _H3M] > 0) == (y[:, _H3M] > 0)).sum().item()
        n_rows += x.shape[0]
    if n_rows == 0:
        return float("nan"), float("nan")
    return total_loss / n_rows, correct / n_rows


# ── Core training (DB-free so it is unit-testable) ───────────────────────────

def run_training(
    bundle: DatasetBundle,
    *,
    checkpoint_path: Path = CHECKPOINT_PATH,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    device: torch.device | None = None,
    seed: int = RANDOM_SEED,
) -> FTTransformerModel:
    """Train on ``bundle`` and save the best (lowest val loss) checkpoint.

    Kept independent of the database so the smoke tests can drive it with a
    synthetic :class:`DatasetBundle`. ``main`` supplies the real bundle.
    """
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(bundle.train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(bundle.val, batch_size=batch_size, shuffle=False)

    model = FTTransformerModel(**MODEL_CONFIG).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dir_acc = evaluate_split(model, val_loader, device)
        scheduler.step()

        logger.info(
            "epoch %3d | train_loss %.6f | val_loss %.6f | dir_acc_3M %.4f",
            epoch, train_loss, val_loss, val_dir_acc,
        )
        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.6f} | "
            f"val_loss {val_loss:.6f} | direction_accuracy {val_dir_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {"state_dict": model.state_dict(), "config": MODEL_CONFIG},
                checkpoint_path,
            )
            logger.info("saved new best checkpoint (val_loss %.6f) -> %s", val_loss, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    return model


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from ..models.db import engine

    with Session(engine) as session:
        bundle = build_datasets(session)
    run_training(bundle)
    print(f"best checkpoint written to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
