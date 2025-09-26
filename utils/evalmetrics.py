import os
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score
)

# Function to save  the metrics
def save_metrics_to_npy(metrics, save_dir):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for metric_name, metric_values in metrics.items():
        # Convert each metric list to a NumPy array and save it
        np.save(os.path.join(save_dir, f"{metric_name}.npy"), np.array(metric_values))

# compute dice_coefficient
@torch.no_grad()
def dice_coef(logits, target, smooth=1.):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum()
    return (2*inter + smooth) / (pred.sum() + target.sum() + smooth)


# compute final model metrics
@torch.no_grad()
def classification_metrics(y_true, y_pred):
    # Convert from torch.Tensor to numpy, if needed
    if hasattr(y_true, 'detach'):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, 'detach'):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten and ensure arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Binarize both to uint8
    y_true_bin = (y_true > 0.5).astype(np.uint8)
    y_pred_bin = (y_pred > 0.5).astype(np.uint8)


    # Now we are guaranteed to have 0/1 in both
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    specificity = tn / (tn + fp + 1e-6)

    return acc, specificity, rec, prec, f1, y_true_bin, y_pred_bin



