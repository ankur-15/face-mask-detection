"""
Dataset Loader for Face Mask Detection
Expects directory structure:
    data/
    ├── with_mask/       ← images of faces with masks
    └── without_mask/    ← images of faces without masks
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── Transforms ────────────────────────────────────────────────────────────────

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

INFERENCE_TRANSFORMS = VAL_TRANSFORMS   # alias for detection scripts

CLASS_NAMES = ["with_mask", "without_mask"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ── Dataset ───────────────────────────────────────────────────────────────────

class MaskDataset(Dataset):
    """Custom dataset that reads images from class-named subdirectories."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []   # list of (image_path, label)

        for label_name, label_idx in CLASS_TO_IDX.items():
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(class_dir):
                print(f"[WARNING] Directory not found: {class_dir}")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label_idx)
                    )

        print(f"[Dataset] Loaded {len(self.samples)} samples from '{root_dir}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=4):
    """
    Split the dataset into train/val and return DataLoaders.

    Args:
        data_dir    : path to dataset root (contains with_mask/ & without_mask/)
        batch_size  : mini-batch size
        val_split   : fraction of data for validation
        num_workers : parallel data loading workers

    Returns:
        (train_loader, val_loader, class_names)
    """
    from torch.utils.data import random_split

    full_dataset = MaskDataset(data_dir, transform=TRAIN_TRANSFORMS)
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Apply stricter transform to val split
    val_set.dataset.transform = VAL_TRANSFORMS  # note: modifies shared dataset

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"[DataLoader] Train: {train_size} | Val: {val_size}")
    return train_loader, val_loader, CLASS_NAMES
