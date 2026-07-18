from __future__ import annotations

import pytest
import torch
from torch import nn

from app.ml.backbone import FTTransformerBackbone, FTTransformerModel
from app.ml.features import HORIZONS, N_FEATURES


def _small_backbone() -> FTTransformerBackbone:
    return FTTransformerBackbone(
        d_model=32,
        n_heads=4,
        num_layers=1,
        ffn_dim=64,
        dropout=0.0,
    )


def _small_model() -> FTTransformerModel:
    return FTTransformerModel(
        d_model=32,
        n_heads=4,
        num_layers=1,
        ffn_dim=64,
        dropout=0.0,
    )


def test_backbone_returns_cls_embedding_with_pre_norm_encoder():
    backbone = _small_backbone()
    tokens = torch.randn(3, N_FEATURES + 1, 32)

    output = backbone(tokens)

    assert output.shape == (3, 32)
    layer = backbone.encoder.layers[0]
    assert isinstance(layer, nn.TransformerEncoderLayer)
    assert layer.norm_first is True
    assert layer.self_attn.batch_first is True


def test_backbone_rejects_incompatible_token_shape():
    backbone = _small_backbone()

    with pytest.raises(ValueError, match="expected tokens"):
        backbone(torch.randn(2, N_FEATURES + 1, 31))


def test_model_produces_all_horizon_predictions_and_gradients():
    model = _small_model()
    model.eval()
    features = torch.randn(2, N_FEATURES, requires_grad=True)

    predictions = model(features)
    predictions.sum().backward()

    assert predictions.shape == (2, len(HORIZONS))
    assert features.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_model_uses_persona_mask_instead_of_training_dropout(monkeypatch):
    model = _small_model()
    model.train()
    calls: list[tuple[str, object]] = []

    def apply_persona_mask(tokens: torch.Tensor, persona_name: str) -> torch.Tensor:
        calls.append(("persona", persona_name))
        return tokens

    def random_modality_dropout(tokens: torch.Tensor, p: float) -> torch.Tensor:
        calls.append(("dropout", p))
        return tokens

    monkeypatch.setattr(model.modality_mask, "apply_persona_mask", apply_persona_mask)
    monkeypatch.setattr(model.modality_mask, "random_modality_dropout", random_modality_dropout)

    model(torch.randn(1, N_FEATURES), persona_name="value_fundamentalist")

    assert calls == [("persona", "value_fundamentalist")]


def test_model_applies_dropout_only_while_training(monkeypatch):
    model = _small_model()
    calls: list[float] = []

    def random_modality_dropout(tokens: torch.Tensor, p: float) -> torch.Tensor:
        calls.append(p)
        return tokens

    monkeypatch.setattr(model.modality_mask, "random_modality_dropout", random_modality_dropout)

    model.train()
    model(torch.randn(1, N_FEATURES))
    model.eval()
    model(torch.randn(1, N_FEATURES))

    assert calls == [model.modality_dropout_p]


def test_model_rejects_invalid_modality_dropout_probability():
    with pytest.raises(ValueError, match="modality_dropout_p"):
        FTTransformerModel(modality_dropout_p=1.1)
