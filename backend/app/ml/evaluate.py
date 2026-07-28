# Evaluation metrics for the FT-Transformer on the held-out test split (2024+).
#
#   - Load the best checkpoint from backend/app/ml/checkpoints/best_model.pt
#   - Per-horizon table (all len(HORIZONS) horizons): MAE, RMSE, direction accuracy
#   - Per-persona table: run each PERSONA_MODALITIES mask, report 3M direction accuracy
#   - Sharpe-proxy: at non-overlapping ~1M rebalance dates, rank tickers by
#     predicted 1M return, go long the top-2, annualize the realized Sharpe
#
# Entry point: run from the backend/ directory: python -m app.ml.evaluate
#
# Depends on: dataset.py, backbone.py (FTTransformerModel), features.py, labels.py

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sqlalchemy.orm import Session

from .backbone import FTTransformerModel
from .dataset import TRADING_DAYS_PER_YEAR, AnalystMindDataset, build_datasets
from .features import HORIZONS, PERSONA_MODALITIES
from .labels import LABEL_COLUMNS
from .train import CHECKPOINT_PATH

logger = logging.getLogger(__name__)

# Named horizons, resolved by value so they stay correct if HORIZONS changes.
_H1M = HORIZONS.index(21)   # 1 month  = 21 trading days
_H3M = HORIZONS.index(63)   # 3 months = 63 trading days
_MONTH_TRADING_DAYS = 21
_TOP_K = 2                   # go long the top-2 ranked tickers per rebalance date


# ── Checkpoint loading ───────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> FTTransformerModel:
    """Rebuild the model from the saved config and load its weights (eval mode)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FTTransformerModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# ── Prediction ───────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    model: FTTransformerModel,
    dataset: AnalystMindDataset,
    device: torch.device,
    persona_name: str | None = None,
) -> torch.Tensor:
    """Full-batch forward pass over a dataset, optionally through a persona mask."""
    x = dataset.features.to(device)
    return model(x, persona_name=persona_name).cpu()


# ── Metrics (pure — unit-testable) ───────────────────────────────────────────

def per_horizon_metrics(preds: torch.Tensor, targets: torch.Tensor) -> pd.DataFrame:
    """MAE, RMSE, and direction accuracy for every horizon column."""
    rows = []
    for i, (horizon, column) in enumerate(zip(HORIZONS, LABEL_COLUMNS)):
        err = preds[:, i] - targets[:, i]
        mae = err.abs().mean().item()
        rmse = torch.sqrt((err ** 2).mean()).item()
        dir_acc = ((preds[:, i] > 0) == (targets[:, i] > 0)).float().mean().item()
        rows.append(
            {"horizon_days": horizon, "label": column, "MAE": mae, "RMSE": rmse,
             "direction_acc": dir_acc}
        )
    return pd.DataFrame(rows)


def persona_direction_accuracy(
    model: FTTransformerModel,
    dataset: AnalystMindDataset,
    device: torch.device,
    horizon_idx: int = _H3M,
) -> pd.DataFrame:
    """3M direction accuracy for each persona, run through its modality mask."""
    targets = dataset.labels
    rows = []
    for persona in PERSONA_MODALITIES:
        preds = predict(model, dataset, device, persona_name=persona)
        dir_acc = ((preds[:, horizon_idx] > 0) == (targets[:, horizon_idx] > 0)).float().mean().item()
        rows.append({"persona": persona, "direction_acc_3M": dir_acc})
    return pd.DataFrame(rows)


def sharpe_proxy(
    pred_1m: np.ndarray,
    actual_1m: np.ndarray,
    dates: np.ndarray,
    top_k: int = _TOP_K,
) -> float:
    """Annualized Sharpe of a long-top-``k`` strategy on 1M forward returns.

    Rebalances on **non-overlapping** dates (every ``_MONTH_TRADING_DAYS`` unique
    trading dates) so the held 1M returns don't overlap — overlapping windows
    would inflate the Sharpe through autocorrelation. Returns ``nan`` when there
    are too few rebalance dates or the realized returns have zero variance.
    """
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "pred": pred_1m, "actual": actual_1m})
    unique_dates = np.sort(frame["date"].unique())
    rebalance_dates = unique_dates[::_MONTH_TRADING_DAYS]

    period_returns: list[float] = []
    for date in rebalance_dates:
        day = frame[frame["date"] == date]
        if len(day) < top_k:
            continue  # not enough names to form the book
        top = day.nlargest(top_k, "pred")
        period_returns.append(float(top["actual"].mean()))

    if len(period_returns) < 2:
        return float("nan")
    returns = np.asarray(period_returns)
    std = returns.std(ddof=1)
    if std == 0:
        return float("nan")
    periods_per_year = TRADING_DAYS_PER_YEAR / _MONTH_TRADING_DAYS
    return float(returns.mean() / std * np.sqrt(periods_per_year))


# ── Reporting ────────────────────────────────────────────────────────────────

def _print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string(index=False))


def run_evaluation(
    bundle_test: AnalystMindDataset,
    model: FTTransformerModel,
    device: torch.device,
) -> None:
    """Compute and print all three metric tables for the test split."""
    targets = bundle_test.labels
    preds = predict(model, bundle_test, device)

    horizon_table = per_horizon_metrics(preds, targets)
    _print_table("Per-horizon metrics", horizon_table)

    persona_table = persona_direction_accuracy(model, bundle_test, device)
    _print_table("Per-persona 3M direction accuracy", persona_table)

    dates = bundle_test.index.get_level_values("date").to_numpy()
    sharpe = sharpe_proxy(
        preds[:, _H1M].numpy(),
        targets[:, _H1M].numpy(),
        dates,
    )
    print(f"\nSharpe-proxy (long top-{_TOP_K} by predicted 1M return, annualized): {sharpe:.4f}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"no checkpoint at {CHECKPOINT_PATH} — run `python -m app.ml.train` first"
        )

    from ..models.db import engine

    with Session(engine) as session:
        bundle = build_datasets(session)

    model = load_model(CHECKPOINT_PATH, device)
    run_evaluation(bundle.test, model, device)


if __name__ == "__main__":
    main()
