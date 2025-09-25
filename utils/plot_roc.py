import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(ground_truth_tensor, prediction_tensor):
    """
    Plot ROC curve from two PyTorch tensors.
    
    Parameters:
        ground_truth_tensor (torch.Tensor): Binary tensor of shape [H, W] or [1, H, W]
        prediction_tensor (torch.Tensor): Float tensor with values in [0, 1], same shape as ground_truth_tensor
    """
    # Remove batch/channel dimensions if necessary
    if ground_truth_tensor.dim() == 3:
        ground_truth_tensor = ground_truth_tensor.squeeze()
    if prediction_tensor.dim() == 3:
        prediction_tensor = prediction_tensor.squeeze()

    # Flatten and move to CPU
    gt_flat = ground_truth_tensor.detach().cpu().numpy().flatten()
    pred_flat = prediction_tensor.detach().cpu().numpy().flatten()

    # Sanity check: binarize ground truth if needed
    gt_binary = (gt_flat > 0.5).astype(int)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(gt_binary, pred_flat)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Between Two Image Tensors')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Example usage (assuming you already have these tensors)
# ground_truth_tensor = torch.randint(0, 2, (1, 256, 256)).float()  # Binary mask
# prediction_tensor = torch.rand(1, 256, 256)  # Predicted probability map

# plot_roc_from_tensors(ground_truth_tensor, prediction_tensor)
