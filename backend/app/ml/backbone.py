"""FT-Transformer encoder and multi-horizon return model.

A row of tabular features is treated as a set of feature tokens rather than a
sequence.  Consequently this module deliberately has no positional encoding:
the identity of each token is already represented by the feature-specific
weights in :class:`FeatureTokenizer`.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import N_FEATURES
from .heads import PredictionHead
from .tokenizer import D_MODEL, FeatureTokenizer, ModalityMask


class FTTransformerBackbone(nn.Module):
    """Encode feature tokens and return the contextualised ``[CLS]`` embedding.

    The encoder uses pre-layer normalization, which is generally more stable
    for the relatively small tabular batches used by this project.  Feature
    token positions are fixed by the shared ``FEATURE_ORDER`` contract, but
    they are not temporal positions, so positional embeddings are omitted.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = 8,
        num_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        # ``batch_first`` preserves the (batch, tokens, embedding) convention
        # used by FeatureTokenizer.  No positional encoding is added here.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.d_model = d_model
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the encoder output for the leading ``[CLS]`` token.

        Parameters
        ----------
        tokens:
            Tensor shaped ``(batch, n_features + 1, d_model)``.  Position zero
            must be the ``[CLS]`` token created by :class:`FeatureTokenizer`.
        """
        if tokens.ndim != 3 or tokens.shape[-1] != self.d_model:
            raise ValueError(
                "expected tokens shaped (batch, n_tokens, "
                f"{self.d_model}), got {tuple(tokens.shape)}"
            )

        encoded = self.encoder(tokens)
        return encoded[:, 0, :]


class FTTransformerModel(nn.Module):
    """Predict forward returns for all horizons from one feature snapshot.

    Persona masking is an inference-time policy: it exposes only the feature
    modalities relevant to a persona while retaining ``[CLS]``.  During
    ordinary training, whole modalities are dropped instead to make the shared
    model robust to unavailable data sources.  A requested persona always
    takes precedence over random dropout, including when the model is in
    training mode, so its output remains deterministic with respect to its
    active modality set.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = 8,
        num_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        modality_dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        if not 0.0 <= modality_dropout_p <= 1.0:
            raise ValueError("modality_dropout_p must be in [0, 1]")

        # All modules share the canonical N_FEATURES/FEATURE_ORDER contract.
        self.tokenizer = FeatureTokenizer(n_features=N_FEATURES, d_model=d_model)
        self.modality_mask = ModalityMask(n_features=N_FEATURES)
        self.backbone = FTTransformerBackbone(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.head = PredictionHead(d_model=d_model)
        self.modality_dropout_p = modality_dropout_p

    def forward(self, x: torch.Tensor, persona_name: str | None = None) -> torch.Tensor:
        """Return predicted returns in ``features.HORIZONS`` order.

        ``x`` must use the canonical ``FEATURE_ORDER`` column order.  Passing
        ``persona_name`` raises ``KeyError`` for an unknown persona via
        :class:`ModalityMask`, rather than silently producing a misleading
        all-feature prediction.
        """
        tokens = self.tokenizer(x)

        if persona_name is not None:
            tokens = self.modality_mask.apply_persona_mask(tokens, persona_name)
        elif self.training:
            tokens = self.modality_mask.random_modality_dropout(
                tokens,
                p=self.modality_dropout_p,
            )

        cls_embedding = self.backbone(tokens)
        return self.head(cls_embedding)
