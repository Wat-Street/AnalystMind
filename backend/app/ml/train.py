# ASSIGNED TO: Preston
#
# Task: Implement the training loop.
#
# What to implement:
#   - AdamW optimizer, lr=1e-4, weight_decay=1e-5
#   - CosineAnnealingLR scheduler
#   - Loss: MSELoss summed over all 4 horizon heads
#   - Modality dropout active during training (passed to dataset/tokenizer)
#   - Early stopping on val loss, patience=10 epochs
#   - Save best checkpoint to: backend/app/ml/checkpoints/best_model.pt
#   - Print per-epoch: train_loss, val_loss, direction_accuracy (sign match on 3M head)
#
# Entry point: python -m backend.app.ml.train
#
# Depends on: dataset.py, labels.py, backbone.py (FTTransformerModel)
