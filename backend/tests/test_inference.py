from __future__ import annotations

import numpy as np
import pytest
import torch

from app.ml.features import FEATURE_ORDER, HORIZONS
from app.ml.inference import PersonaInference
from app.schemas.persona import PersonaOutput


class _FixedModel:
    def __init__(self, predictions: list[float]) -> None:
        self.predictions = torch.tensor([predictions], dtype=torch.float32)
        self.calls: list[tuple[torch.Tensor, str | None]] = []

    def __call__(self, features: torch.Tensor, persona_name: str | None = None) -> torch.Tensor:
        self.calls.append((features, persona_name))
        return self.predictions


def _service(predictions: list[float]) -> tuple[PersonaInference, _FixedModel]:
    service = PersonaInference.__new__(PersonaInference)
    model = _FixedModel(predictions)
    service.model = model
    service.device = torch.device("cpu")
    return service, model


def _output(persona: str, price_target: float, confidence: float = 1.0) -> PersonaOutput:
    return PersonaOutput(
        ticker="AAPL",
        persona_name=persona,
        pt_base=price_target,
        confidence=confidence,
    )


def test_run_persona_uses_canonical_features_and_three_month_return(monkeypatch):
    predictions = [0.01, 0.02, 0.10, 0.04, -0.02]
    service, model = _service(predictions)
    feature_vector = torch.arange(len(FEATURE_ORDER), dtype=torch.float32)
    monkeypatch.setattr(
        service,
        "_latest_snapshot",
        lambda session, ticker: (100.0, feature_vector),
    )

    output = service.run_persona(" aapl ", "technical_analyst", object())

    assert output.ticker == "AAPL"
    assert output.pt_base == pytest.approx(110.0)
    assert output.pt_bull == pytest.approx(112.0)
    assert output.pt_bear == pytest.approx(98.0)
    assert output.confidence == pytest.approx(1.0)
    assert model.calls[0][1] == "technical_analyst"
    torch.testing.assert_close(model.calls[0][0], feature_vector.unsqueeze(0))


def test_run_persona_rejects_wrong_model_output_shape(monkeypatch):
    service, _ = _service([0.01, 0.02, 0.03, 0.04])
    monkeypatch.setattr(
        service,
        "_latest_snapshot",
        lambda session, ticker: (100.0, torch.zeros(len(FEATURE_ORDER))),
    )

    with pytest.raises(RuntimeError, match=r"expected \(1, 5\)"):
        service.run_persona("AAPL", "technical_analyst", object())


def test_consensus_downweights_a_far_outlier():
    service, _ = _service([0.0] * len(HORIZONS))
    outputs = [
        _output("value_fundamentalist", 100.0),
        _output("growth_visionary", 102.0),
        _output("quant_momentum", 104.0),
        _output("technical_analyst", 200.0),
    ]

    consensus = service.aggregate_consensus("aapl", outputs)

    sigma = np.std([100.0, 102.0, 104.0, 200.0])
    expected = (100.0 + 102.0 + 104.0 + 0.30 * 200.0) / 3.30
    assert consensus.ticker == "AAPL"
    assert consensus.consensus_pt == pytest.approx(expected)
    assert consensus.band_low == pytest.approx(expected - sigma)
    assert consensus.band_high == pytest.approx(expected + sigma)
    assert consensus.outlier_persona == "technical_analyst"
    assert consensus.outlier_pt == pytest.approx(200.0)
    assert consensus.conviction_score == pytest.approx(max(0.0, min(1.0, 1.0 - sigma / expected)))


def test_consensus_falls_back_to_unweighted_mean_when_confidence_is_zero():
    service, _ = _service([0.0] * len(HORIZONS))
    outputs = [
        _output("value_fundamentalist", 90.0, confidence=0.0),
        _output("growth_visionary", 110.0, confidence=0.0),
    ]

    consensus = service.aggregate_consensus("AAPL", outputs)

    assert consensus.consensus_pt == pytest.approx(100.0)
    assert 0.0 <= consensus.conviction_score <= 1.0


def test_consensus_requires_persona_outputs():
    service, _ = _service([0.0] * len(HORIZONS))

    with pytest.raises(ValueError, match="at least one persona output"):
        service.aggregate_consensus("AAPL", [])
