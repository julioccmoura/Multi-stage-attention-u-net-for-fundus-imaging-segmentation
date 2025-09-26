import numpy as np
import os
from pathlib import Path

import torch
from torch.utils.data import  DataLoader
from torchsummary import summary

from model import AttentionUNet
from utils.dataset import SegmentationDataset
from utils.augmentation import train_transform, val_transform
from utils.train import train_one_stage
from utils.losses import BCELoss, TverskyLoss, FocalTverskyLoss
from utils.show_predictions import show_predictions
from utils.evalmetrics import save_metrics_to_npy
from utils.plot_curves import training_curves, evaluate_and_plot_roc

# Repro
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)

def main(images_path, masks_path, save_model_path, output_dir, n_epochs, patience):
    # Check if  cuda is available o.t.w runs on cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare the output dir for saving
    os.makedirs(output_dir, exist_ok=True)
    curves_dir = os.path.join(output_dir, "curves")
    preds_dir  = os.path.join(output_dir, "preds")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    
    # Prepare the data for training and validation
    train_set = SegmentationDataset(images_path, masks_path, transform=train_transform)
    val_set   = SegmentationDataset(images_path, masks_path, transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    
    # Defines the model
    model = AttentionUNet(img_ch=3, output_ch=1).to(device)
    summary(model, input_size=(3,256,256))
    
    # Defines the stages
    stages = [
        ("BCE",          BCELoss(),                         n_epochs, patience, 1e-3),
        ("Tversky",      TverskyLoss(alpha=0.7, beta=0.3),               n_epochs, patience, 7e-4),
        ("FocalTversky", FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75), n_epochs, patience, 5e-4),
    ]
    best_dices = {}
    
    for stage_name, criterion, n_epochs, patience, lr in stages:
        # Load best so far (if exists)
        if os.path.exists(save_model_path):
            model.load_state_dict(torch.load(save_model_path, map_location=device))
        
        # Defines the optimizer (Adam)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        metrics = train_one_stage(
            model, train_loader, val_loader,
            optimizer, criterion,
            save_path=save_model_path,
            n_epochs=n_epochs, patience=patience, stage=stage_name
        )
        
        best_dices[stage_name] = metrics["best_dice"]
        
        # Reload stage best for fair visualization
        model.load_state_dict(torch.load(save_model_path, map_location=device))
        
        # Prepare the directories to save according different stages
        stage_pred_dir = os.path.join(preds_dir, stage_name)
        stage_metrics_dir = os.path.join(metrics_dir, stage_name)
        stage_curves_dir = os.path.join(curves_dir, stage_name)
        
        # Show and save the predictions and save the metrics
        show_predictions(model, val_set, device, num=6, stage_name=stage_name, save_dir=stage_pred_dir)
        save_metrics_to_npy(metrics, save_dir=stage_metrics_dir)
        
        # Show and save the curves
        training_curves(stage_metrics_dir, stage_curves_dir)
        evaluate_and_plot_roc(model, val_set, device, stage_curves_dir)


# Main settings
dataset_images = r"../DRIONS-DB/images/"
dataset_masks = r"../DRIONS-DB/masks/"
save_model_path = r"outputs/DRIONS-DB/AttUnet_DRIONS.pth"
output_dir = r"outputs/DRIONS-DB"
n_epochs = 50
patience = 10


if __name__ == "__main__":
    main(dataset_images, dataset_masks, save_model_path, output_dir, n_epochs, patience)







