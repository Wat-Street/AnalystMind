# ASSIGNED TO: Preston
#
# Task: Implement evaluation metrics on the held-out test set.
#
# What to implement:
#   - Load best checkpoint from backend/app/ml/checkpoints/best_model.pt
#   - Run on test split (2024+) from dataset.py
#   - Print a table per horizon (1M, 3M, 6M, 12M):
#       MAE, RMSE, direction accuracy (sign(pred) == sign(actual))
#   - Per-persona accuracy table: run each PERSONA_MODALITIES mask independently,
#       report direction accuracy for each persona on the 3M horizon
#   - Sharpe-proxy: at each date, rank 5 tickers by predicted 1M return,
#       go long top-2, compute realized return, report annualized Sharpe
#
# Entry point: python -m backend.app.ml.evaluate
#
# Depends on: dataset.py, backbone.py (FTTransformerModel), features.py (PERSONA_MODALITIES)
