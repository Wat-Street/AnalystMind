# ASSIGNED TO: Kiana
#
# Task: Compute forward return labels from the ohlcv table.
#
# What to implement:
#   - For each (ticker, date), compute forward returns at all 4 horizons:
#       label = (adj_close[t + N] - adj_close[t]) / adj_close[t]
#       for N in HORIZONS (21, 63, 126, 252 trading days)
#   - Return a DataFrame indexed by (ticker, date) with columns:
#       return_1m, return_3m, return_6m, return_12m
#   - Rows near the end of history that have no future price should be dropped.
#
# Depends on: features.py (HORIZONS), ohlcv DB table
