import cv2
import os
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', 'tif'))])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Read the image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image and mask
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # Apply transformations in casa of augmentation 
        if self.transform:
            augmented = self.transform(image=image, mask=mask)  # Pass both image and mask
            image = augmented["image"]
            mask = augmented["mask"]
            mask = (mask / 255).unsqueeze(0).float()  # Convert mask to float and normalize
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.

        return image, mask
