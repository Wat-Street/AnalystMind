# ASSIGNED TO: Ken
#
# Task: Implement the FT-Transformer Feature Tokenizer and modality masking logic.
#
# What to implement:
#
#   class FeatureTokenizer(nn.Module):
#     - Learnable per-feature linear projection: x_i (scalar) → e_i (d_model,)
#       implemented as nn.Parameter weight (N_features, d_model) + bias (N_features, d_model)
#     - Prepend a learned [CLS] token
#     - Input:  (batch, N_features)
#     - Output: (batch, N_features + 1, d_model)
#
#   class ModalityMask:
#     - apply_persona_mask(tokens, persona_name) → zeros out token positions not in
#       that persona's modality group (use PERSONA_MODALITIES from features.py)
#     - random_modality_dropout(tokens, p=0.3) → during training, randomly zero out
#       entire modality blocks independently with probability p each
#
# Depends on: features.py (FEATURE_ORDER, MODALITY_GROUPS, PERSONA_MODALITIES, N_FEATURES)
# Coordinate with Kiana: FEATURE_ORDER index positions must match dataset.py column order.
