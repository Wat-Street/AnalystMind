# ASSIGNED TO: Afreed
#
# Task: Implement the multi-horizon prediction head.
#
# What to implement:
#
#   class PredictionHead(nn.Module):
#     - Two-layer MLP: d_model → 128 → len(HORIZONS)  [outputs 4 return predictions]
#     - Input:  (batch, d_model)   [CLS token from backbone]
#     - Output: (batch, 4)         [predicted returns for 1M, 3M, 6M, 12M]
#     - Use ReLU activation between layers, no activation on output
#
# Depends on: features.py (HORIZONS), backbone.py uses this in FTTransformerModel
