import albumentations as A
from albumentations.pytorch import ToTensorV2

# Train transformations
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
    A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
    ToTensorV2(transpose_mask=True)
])

# Validation transformations
val_transform = A.Compose([
    A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
    ToTensorV2(transpose_mask=True)
])
