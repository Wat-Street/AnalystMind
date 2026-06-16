# ASSIGNED TO: Kiana
#
# Task: Build the ML dataset by joining all ingestion tables into per-(ticker, date) snapshots.
#
# What to implement:
#   - Query ohlcv, fundamentals, insider_activity, sentiment_scores, macro_series from DB
#   - Join on (ticker, date) — one row = one training example
#   - Impute nulls with column median; add binary missing-indicator columns
#   - Time-based train/val/test split (train ≤ 2022, val 2023, test 2024+)
#   - Return a torch.utils.data.Dataset yielding (feature_tensor, label_tensor)
#
# Depends on: features.py (FEATURE_ORDER), labels.py (forward return labels)
# Coordinate with Ken on FEATURE_ORDER before writing any column-selection logic.
