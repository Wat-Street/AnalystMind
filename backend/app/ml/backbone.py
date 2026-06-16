# ASSIGNED TO: Aadesh
#
# Task: Implement the FT-Transformer encoder backbone and top-level model wrapper.
#
# What to implement:
#
#   class FTTransformerBackbone(nn.Module):
#     - Standard Transformer encoder, L=3 layers, d_model=192, n_heads=8, ffn_dim=512, dropout=0.1
#     - Pre-norm style (norm_first=True in nn.TransformerEncoderLayer)
#     - No positional encoding — tabular features have no sequence order
#     - Input:  (batch, N_tokens, d_model)   [N_tokens = N_features + 1 for CLS]
#     - Output: (batch, d_model)              [CLS token at index 0 of sequence]
#
#   class FTTransformerModel(nn.Module):
#     - Wraps FeatureTokenizer + FTTransformerBackbone + PredictionHead into one callable
#     - forward(x, persona_name=None) → (batch, 4) predicted returns
#       If persona_name is given, applies persona mask before backbone.
#       If training, applies random_modality_dropout instead.
#
# Depends on: tokenizer.py (FeatureTokenizer, ModalityMask), heads.py (PredictionHead)
# Add torch and rtdl to backend/requirements.txt.
# You also own inference.py — backbone + inference are tightly coupled, makes sense to keep together.
