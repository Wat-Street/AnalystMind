from __future__ import annotations

import pytest
import torch

from app.ml.features import MODALITY_GROUPS, N_FEATURES, PERSONA_MODALITIES
from app.ml.tokenizer import D_MODEL, FeatureTokenizer, ModalityMask


def test_tokenizer_output_shape_and_cls_position():
    tok = FeatureTokenizer()
    x = torch.randn(4, N_FEATURES)
    out = tok(x)
    assert out.shape == (4, N_FEATURES + 1, D_MODEL)
    assert torch.allclose(out[:, 0], tok.cls_token.expand(4, D_MODEL))


def test_tokenizer_is_affine_per_feature():
    tok = FeatureTokenizer()
    x = torch.randn(2, N_FEATURES)
    out = tok(x)
    # Feature token i (position i+1) = bias_i + x_i * weight_i.
    expected = tok.bias + x.unsqueeze(-1) * tok.weight
    assert torch.allclose(out[:, 1:], expected, atol=1e-6)


def test_tokenizer_zero_input_returns_bias():
    tok = FeatureTokenizer()
    out = tok(torch.zeros(1, N_FEATURES))
    assert torch.allclose(out[:, 1:], tok.bias.unsqueeze(0), atol=1e-6)


def test_tokenizer_rejects_wrong_feature_count():
    tok = FeatureTokenizer()
    with pytest.raises(ValueError):
        tok(torch.randn(3, N_FEATURES + 1))


def test_tokenizer_gradients_flow():
    tok = FeatureTokenizer()
    out = tok(torch.randn(2, N_FEATURES))
    out.sum().backward()
    assert tok.weight.grad is not None
    assert tok.cls_token.grad is not None


# modality masking
def _tokens(batch: int = 3) -> torch.Tensor:
    return torch.ones(batch, N_FEATURES + 1, D_MODEL)


def test_persona_mask_keeps_only_persona_modalities():
    mask = ModalityMask()
    persona = "value_fundamentalist"
    kept_features = {
        idx
        for modality in PERSONA_MODALITIES[persona]
        for idx in MODALITY_GROUPS[modality]
    }
    out = mask.apply_persona_mask(_tokens(), persona)

    assert (out[:, 0] == 1).all()  # CLS always there
    for feat_idx in range(N_FEATURES):
        token_alive = bool((out[:, feat_idx + 1] != 0).any())
        assert token_alive == (feat_idx in kept_features)


def test_persona_mask_unknown_persona_raises():
    mask = ModalityMask()
    with pytest.raises(KeyError):
        mask.apply_persona_mask(_tokens(), "nope")


def test_every_persona_resolves():
    mask = ModalityMask()
    for persona in PERSONA_MODALITIES:
        out = mask.apply_persona_mask(_tokens(1), persona)
        assert (out[:, 0] == 1).all()
        assert (out != 0).any()  # a persona always keeps at least one modality


# modality dropout
def test_dropout_p_zero_is_noop():
    mask = ModalityMask()
    t = _tokens()
    assert torch.equal(mask.random_modality_dropout(t, p=0.0), t)


def test_dropout_p_one_zeros_all_features_but_keeps_cls():
    mask = ModalityMask()
    out = mask.random_modality_dropout(_tokens(), p=1.0)
    assert (out[:, 0] == 1).all()      # CLS never dropped
    assert (out[:, 1:] == 0).all()     # every modality dropped


def test_dropout_keeps_whole_modality_blocks_intact():
    torch.manual_seed(0)
    mask = ModalityMask()
    out = mask.random_modality_dropout(_tokens(8), p=0.5)
    # Within a sample, a modality's feature tokens are all kept or all zero
    for sample in range(out.shape[0]):
        for feats in MODALITY_GROUPS.values():
            alive = [bool((out[sample, i + 1] != 0).any()) for i in feats]
            assert len(set(alive)) == 1


def test_dropout_invalid_p_raises():
    mask = ModalityMask()
    with pytest.raises(ValueError):
        mask.random_modality_dropout(_tokens(), p=1.5)
