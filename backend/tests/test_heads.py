from __future__ import annotations

import torch
from torch import nn

from app.ml.features import HORIZONS
from app.ml.heads import PredictionHead


def test_prediction_head_output_shape():
    head = PredictionHead()
    output = head(torch.randn(4, 192))

    assert output.shape == (4, len(HORIZONS))


def test_prediction_head_architecture():
    head = PredictionHead()

    assert isinstance(head.layers[0], nn.Linear)
    assert head.layers[0].in_features == 192
    assert head.layers[0].out_features == 128
    assert isinstance(head.layers[1], nn.ReLU)
    assert isinstance(head.layers[2], nn.Linear)
    assert head.layers[2].in_features == 128
    assert head.layers[2].out_features == len(HORIZONS)


def test_prediction_head_supports_custom_dimensions():
    head = PredictionHead(d_model=64, hidden_dim=32)
    output = head(torch.randn(2, 64))

    assert output.shape == (2, len(HORIZONS))


def test_prediction_head_gradients_flow():
    head = PredictionHead()
    inputs = torch.randn(2, 192, requires_grad=True)

    head(inputs).sum().backward()

    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in head.parameters())
