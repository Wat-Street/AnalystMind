"""Smoke tests for the training loop and evaluation metrics.

All synthetic — no network, no DB. A tiny in-memory DatasetBundle drives the
DB-free cores (`run_training`, the metric helpers) so the loop and metrics are
verified without a populated Postgres.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from app.ml import evaluate, train
from app.ml.dataset import AnalystMindDataset, DatasetBundle
from app.ml.evaluate import per_horizon_metrics, persona_direction_accuracy, sharpe_proxy
from app.ml.features import FEATURE_ORDER, HORIZONS, PERSONA_MODALITIES
from app.ml.labels import LABEL_COLUMNS
from app.ml.train import direction_accuracy, multi_horizon_mse, run_training

DEVICE = torch.device("cpu")

# Keep the model tiny so the loop runs fast in tests.
_SMALL_CONFIG = {
    "d_model": 32,
    "n_heads": 4,
    "num_layers": 1,
    "ffn_dim": 64,
    "dropout": 0.0,
    "modality_dropout_p": 0.3,
}


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_dataset(n_tickers: int, n_dates: int, seed: int = 0) -> AnalystMindDataset:
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_tickers)]
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    index = pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])
    n = len(index)
    features = pd.DataFrame(
        rng.standard_normal((n, len(FEATURE_ORDER))).astype(np.float32),
        index=index, columns=list(FEATURE_ORDER),
    )
    labels = pd.DataFrame(
        rng.standard_normal((n, len(HORIZONS))).astype(np.float32) * 0.05,
        index=index, columns=list(LABEL_COLUMNS),
    )
    return AnalystMindDataset(features, labels)


@pytest.fixture
def bundle() -> DatasetBundle:
    return DatasetBundle(
        train=make_dataset(4, 30, seed=1),
        val=make_dataset(4, 10, seed=2),
        test=make_dataset(5, 50, seed=3),
        feature_medians=pd.Series(0.0, index=list(FEATURE_ORDER)),
    )


# ── train.py helpers ─────────────────────────────────────────────────────────

def test_multi_horizon_mse_sums_over_horizons():
    preds = torch.zeros(2, len(HORIZONS))
    targets = torch.ones(2, len(HORIZONS))
    # each horizon contributes (0-1)^2 = 1, summed over horizons, mean over batch
    assert multi_horizon_mse(preds, targets).item() == pytest.approx(float(len(HORIZONS)))


def test_direction_accuracy_uses_gt_zero():
    preds = torch.tensor([[1.0], [-1.0], [0.0]])
    targets = torch.tensor([[2.0], [3.0], [4.0]])
    # up/up=hit, down/up=miss, flat(0 not >0)/up=miss -> 1/3
    assert direction_accuracy(preds, targets, 0) == pytest.approx(1 / 3)


# ── training loop ────────────────────────────────────────────────────────────

def test_run_training_saves_checkpoint_and_reloads(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODEL_CONFIG", _SMALL_CONFIG)
    ckpt = tmp_path / "best_model.pt"

    run_training(
        bundle, checkpoint_path=ckpt, max_epochs=3, patience=5,
        batch_size=16, device=DEVICE, seed=0,
    )
    assert ckpt.exists()

    # config round-trips and the reloaded model is deterministic in eval mode
    model = evaluate.load_model(ckpt, DEVICE)
    preds_a = evaluate.predict(model, bundle.test, DEVICE)
    preds_b = evaluate.predict(model, bundle.test, DEVICE)
    assert preds_a.shape == (len(bundle.test), len(HORIZONS))
    assert torch.isfinite(preds_a).all()
    torch.testing.assert_close(preds_a, preds_b)


# ── evaluation metrics ───────────────────────────────────────────────────────

def test_per_horizon_metrics_shape_and_bounds():
    n = 20
    preds = torch.randn(n, len(HORIZONS))
    targets = torch.randn(n, len(HORIZONS))
    table = per_horizon_metrics(preds, targets)

    assert len(table) == len(HORIZONS)
    assert list(table["horizon_days"]) == HORIZONS
    assert (table["MAE"] >= 0).all() and (table["RMSE"] >= 0).all()
    assert table["direction_acc"].between(0.0, 1.0).all()


def test_persona_direction_accuracy_covers_all_personas(bundle):
    model = evaluate.FTTransformerModel(**_SMALL_CONFIG).to(DEVICE)
    model.eval()
    table = persona_direction_accuracy(model, bundle.test, DEVICE)

    assert set(table["persona"]) == set(PERSONA_MODALITIES)
    assert table["direction_acc_3M"].between(0.0, 1.0).all()


def test_sharpe_proxy_finite_on_multi_date_frame():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=60)
    tickers = 5
    all_dates = np.repeat(dates.to_numpy(), tickers)
    pred = rng.standard_normal(len(all_dates))
    actual = rng.standard_normal(len(all_dates)) * 0.05
    result = sharpe_proxy(pred, actual, all_dates)
    assert np.isfinite(result)


def test_sharpe_proxy_degenerate_cases():
    # single rebalance date -> too few periods -> nan
    dates = np.repeat(pd.Timestamp("2024-01-01"), 5)
    assert np.isnan(sharpe_proxy(np.arange(5.0), np.arange(5.0), dates))

    # zero-variance realized returns across rebalance dates -> nan
    dates = pd.bdate_range("2024-01-01", periods=60)
    all_dates = np.repeat(dates.to_numpy(), 3)
    pred = np.tile([3.0, 2.0, 1.0], 60)
    actual = np.zeros(len(all_dates))
    assert np.isnan(sharpe_proxy(pred, actual, all_dates))
