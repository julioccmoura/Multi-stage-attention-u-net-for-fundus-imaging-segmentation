import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score
)

import torch
from torch.utils.data import  DataLoader

from utils.evalmetrics import classification_metrics


def training_curves(metrics_dir, save_dir):
    files = os.listdir(metrics_dir)
    metrics = {}
    
    # reads all the files in the metrics dir as dictionary
    for file in files:
        if file.endswith(".npy"):
            var_name = os.path.splitext(file)[0]
            file_path = os.path.join(metrics_dir, file)
            metrics[var_name] = np.load(file_path)
            
    # estabilish the number of epochs 
    epochs = np.arange(1,len(metrics["train_dice"])+1)
    
    # chose the data to plot (in this case train and test dice and loss)
    train_dice = metrics["train_dice"]      
    val_dice = metrics["val_dice"]        
    
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Dice Coefficient (left y-axis)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dice Coefficient', color='brown')
    ax1.plot(epochs, train_dice, 'r', linestyle="--", label='Training Dice Coefficient')
    ax1.plot(epochs, val_dice, 'r' , label='Validation Dice Coefficient')
    ax1.tick_params(axis='y', labelcolor='brown')
    ax1.set_ylim(0, 1)
    
    # Create second y-axis for Loss
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='blue')
    ax2.plot(epochs, train_loss, 'b', linestyle='--', label='Training Loss')
    ax2.plot(epochs, val_loss, 'b', label='Validation Loss')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, max(max(train_loss), max(val_loss)) + 0.5)
    
    # Title and Legends
    plt.title('Training and Validation Dice Coefficient and Loss over Epochs')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Save the plot if a save directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    
    
    plt.tight_layout()
    plt.show()
    
@torch.no_grad()
def evaluate_and_plot_roc(model, dataset, device, save_dir):

    model.eval()
    all_probs = []
    all_labels = []

    for x, y in DataLoader(dataset, batch_size=1):
        x = x.to(device)
        out = model(x)
        probs = torch.sigmoid(out).cpu().numpy().flatten()
        labels = y.cpu().numpy().flatten()

        all_probs.append(probs)
        all_labels.append(labels)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    # Ensure labels are binary (0 or 1)
    labels = (labels > 0.5).astype(np.uint8)

   
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    
    acc, spec, sens, prec, f1, y_true, y_pred = classification_metrics(labels, probs)

    plt.figure(figsize=(5, 5), dpi=120)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
    plt.show()
    
    # Print final evaluation metrics
    print("\n📊 Final Validation Metrics:")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Specificity:{spec:.4f}")
    print(f"Sensitivity:{sens:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"AUC:        {auc:.4f}")

    
    
