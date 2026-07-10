# ASSIGNED TO: Ken
#
# FT-Transformer feature tokenizer + modality masking.
#
#   FeatureTokenizer   scalar features -> per-feature embeddings + [CLS]
#   ModalityMask       persona masking (inference) and modality dropout (training)

from __future__ import annotations

import math

import torch
from torch import nn

from .features import MODALITY_GROUPS, N_FEATURES, PERSONA_MODALITIES

D_MODEL = 192  # dimension


class FeatureTokenizer(nn.Module):
    """Tokenize scalars into embeddings.

    Each numerical feature x_j is projected like this: e_j = b_j + x_j * W_j
    Add [CLS] token at position 0.

    Input:  (batch, N_features)
    Output: (batch, N_features + 1, d_model)   [CLS at index 0]
    """

    def __init__(self, n_features: int = N_FEATURES, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        # One (d_model,) weight and bias per feature.
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        self.cls_token = nn.Parameter(torch.empty(d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.d_model)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.cls_token, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"expected input (batch, {self.n_features}), got {tuple(x.shape)}"
            )
        #(batch, n_features, d_model): broadcast the scalar x over the d_model axis so it matches
        tokens = self.bias + x.unsqueeze(-1) * self.weight
        cls = self.cls_token.expand(x.shape[0], 1, self.d_model) #virtual copy
        return torch.cat([cls, tokens], dim=1) #put to front


class ModalityMask(nn.Module):
    """Zero out feature tokens by modality, for persona focus or training dropout.

    [CLS] token is never masked. Feature token at position i+1 = feature index i.
    
    **Masks are held as buffers so they follow the module across devices.
    """

    def __init__(
        self,
        persona_modalities: dict[str, list[str]] = PERSONA_MODALITIES,
        modality_groups: dict[str, list[int]] = MODALITY_GROUPS,
        n_features: int = N_FEATURES,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_tokens = n_features + 1  # + CLS
        self.modality_names = list(modality_groups)
        self.persona_names = list(persona_modalities)
        self._persona_row = {name: i for i, name in enumerate(self.persona_names)}

        # membership[m, t] == 1 if token t belongs to modality m (CLS col stays 0)
        membership = torch.zeros(len(self.modality_names), self.n_tokens)
        for m, name in enumerate(self.modality_names):
            for feat_idx in modality_groups[name]:
                membership[m, feat_idx + 1] = 1.0
        self.register_buffer("membership", membership, persistent=False)

        # persona_mask[p, t] == 1 if persona p keeps token t
        persona_mask = torch.zeros(len(self.persona_names), self.n_tokens)
        for p, name in enumerate(self.persona_names):
            persona_mask[p, 0] = 1.0  # always keep CLS
            for modality in persona_modalities[name]:
                m = self.modality_names.index(modality)
                persona_mask[p] = torch.maximum(persona_mask[p], membership[m])
        self.register_buffer("persona_mask", persona_mask, persistent=False)

    def apply_persona_mask(self, tokens: torch.Tensor, persona_name: str) -> torch.Tensor:
        """Zero every token outside persona_name's modality group"""
        if persona_name not in self._persona_row:
            raise KeyError(f"unknown persona")
        mask = self.persona_mask[self._persona_row[persona_name]]  # (n_tokens,)
        return tokens * mask.to(tokens.dtype).view(1, -1, 1)

    def random_modality_dropout(self, tokens: torch.Tensor, p: float = 0.3) -> torch.Tensor:
        """Independently hide whole modalities during training with probability p, prevent overfitting

        dropped tokens are zeroed. CLS is never dropped
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if p == 0.0:
            return tokens
        batch = tokens.shape[0]
        # (batch, n_modalities): 1 where a modality is dropped for that sample
        dropped = (torch.rand(batch, len(self.modality_names), device=tokens.device) < p).to(tokens.dtype)
        #(batch, n_tokens): > 0 where any dropped modality covers the token
        covered = dropped @ self.membership.to(tokens.dtype)
        keep = (covered == 0).to(tokens.dtype)
        return tokens * keep.unsqueeze(-1)
