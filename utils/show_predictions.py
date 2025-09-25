import os
import numpy as np
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def show_predictions(model, dataset, device, num=6, stage_name="", save_dir=None):
    model.eval()
    idxs = np.linspace(0, len(dataset)-1, num=min(num, len(dataset)), dtype=int)

    for i, idx in enumerate(idxs):
        # Load image and mask
        img, mask = dataset[idx]
        inp = img.unsqueeze(0).to(device)

        # Model prediction
        prob = torch.sigmoid(model(inp))[0, 0].cpu().numpy()
        pred = (prob > 0.5).astype(np.uint8)

        # Convert tensors to NumPy
        img_np = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        mask_np = mask[0].cpu().numpy()              # (H, W)

        # Plotting (subplots with 3 images)
        plt.figure(figsize=(12, 3), dpi=150)
        plt.subplot(1, 3, 1); plt.imshow(img_np); plt.axis('off'); plt.title('Image')
        plt.subplot(1, 3, 2); plt.imshow(mask_np, cmap='gray'); plt.axis('off'); plt.title('Ground Truth')
        plt.subplot(1, 3, 3); plt.imshow(pred, cmap='gray'); plt.axis('off'); plt.title('Prediction')
        plt.suptitle(f"{stage_name} – Sample {i+1}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            # Save the subplots figure
            fig_path = os.path.join(save_dir, f"{stage_name}_sample_{i+1}.png")
            plt.savefig(fig_path, bbox_inches='tight', dpi=300)

            # Save each individual image as a figure
            img_fig_path = os.path.join(save_dir, f"{stage_name}_sample_{i+1}_img.png")
            mask_fig_path = os.path.join(save_dir, f"{stage_name}_sample_{i+1}_mask.png")
            pred_fig_path = os.path.join(save_dir, f"{stage_name}_sample_{i+1}_pred.png")

            # Save individual images
            plt.figure(figsize=(6, 6), dpi=150)
            plt.imshow(img_np); plt.axis('off')
            plt.savefig(img_fig_path, bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure(figsize=(6, 6), dpi=150)
            plt.imshow(mask_np, cmap='gray'); plt.axis('off')
            plt.savefig(mask_fig_path, bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure(figsize=(6, 6), dpi=150)
            plt.imshow(pred, cmap='gray'); plt.axis('off')
            plt.savefig(pred_fig_path, bbox_inches='tight', dpi=300)
            plt.close()

        plt.show()
