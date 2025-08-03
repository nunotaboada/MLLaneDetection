import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

class LaneDataset(Dataset):
    """!
    @brief A custom dataset for lane detection.

    This class loads images and their corresponding masks, applies optional data augmentations,
    and returns them as PyTorch tensors for training or validation.

    @param image_dir (str): Directory containing the input images.
    @param mask_dir (str): Directory containing the ground truth masks.
    @param transform (albumentations.Compose, optional): Data augmentation pipeline (default: None).
    @param num_augmentations (int, optional): Number of augmented versions per image (default: 2).
    """
    def __init__(self, image_dir, mask_dir, transform=None, num_augmentations=2):
        self.image_dir = image_dir  #!< Directory with input images.
        self.mask_dir = mask_dir  #!< Directory with ground truth masks.
        self.transform = transform  #!< Optional augmentation pipeline.
        self.num_augmentations = num_augmentations  #!< Number of augmentations per image.
        self.images = os.listdir(image_dir)  #!< List of image filenames.
        self.total_samples = len(self.images) * (1 + self.num_augmentations if transform else 1)  #!< Total dataset size.

        # Base transformation to ensure fixed size without augmentations
        self.base_transform = A.Compose([
            A.Resize(height=128, width=256),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ])

    def __len__(self):
        """!
        @brief Returns the total number of samples in the dataset.

        @return int: Total number of samples, including augmentations.
        """
        return self.total_samples

    def __getitem__(self, index):
        """!
        @brief Retrieves an image and its corresponding mask by index.

        Loads an image and mask, applies augmentations if specified, and returns them as tensors.

        @param index (int): Index of the sample to retrieve.
        @return tuple: A tuple containing the image tensor ([C, H, W]) and mask tensor ([1, H, W]).
        """
        samples_per_image = (1 + self.num_augmentations) if self.transform else 1
        img_idx = index // samples_per_image
        sample_idx = index % samples_per_image
        
        img_path = os.path.join(self.image_dir, self.images[img_idx])
        mask_path = os.path.join(self.mask_dir, self.images[img_idx].replace(".png", "_label.png"))
        
        # Load as NumPy arrays
        image = np.array(Image.open(img_path).convert("RGB"))  # [H, W, 3]
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # [H, W]
        mask[mask == 255.0] = 1.0
        
        # Apply transformations
        if self.transform is not None and sample_idx > 0:
            # Apply augmentations for transformed versions
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]  # [C, H, W]
            mask = augmentations["mask"]    # [1, H, W] or [H, W]
        else:
            # Apply base transformation for original image
            augmentations = self.base_transform(image=image, mask=mask)
            image = augmentations["image"]  # [C, H, W]
            mask = augmentations["mask"]    # [1, H, W]
        
        # Ensure correct mask format
        if len(mask.shape) == 2:  # [H, W] -> [1, H, W]
            mask = mask.unsqueeze(0)
        mask = (mask > 0.5).float()
        
        return image, mask

train_transforms = A.Compose([
    A.Resize(height=144, width=256, interpolation=cv2.INTER_CUBIC),  #!< Resize images to fixed size.
    A.HorizontalFlip(p=0.5),  #!< Randomly flip images horizontally.
    A.RandomBrightnessContrast(p=0.5),  #!< Adjust brightness and contrast randomly.
    A.RandomGamma(p=0.5),  #!< Adjust gamma to enhance lane lines.
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),  #!< Normalize pixel values.
    ToTensorV2(),  #!< Convert to PyTorch tensor.
])

val_transforms = A.Compose([
    A.Resize(height=144, width=256, interpolation=cv2.INTER_CUBIC),  #!< Resize images to fixed size.
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),  #!< Normalize pixel values.
    ToTensorV2(),  #!< Convert to PyTorch tensor.
])

def test():
    """!
    @brief Tests the dataset transformations with a synthetic image and mask.

    Creates a random image and mask, applies training transformations, and prints the shapes and types
    of the resulting tensors for debugging.
    """
    # Create synthetic image and mask as PyTorch tensors
    img_tensor = torch.randint(0, 256, (572, 572, 3), dtype=torch.uint8)  # RGB image
    mask_tensor = torch.randint(0, 2, (572, 572), dtype=torch.uint8)       # Binary mask
    
    print("Data:", img_tensor.shape, img_tensor.dtype)
    print("Targets:", mask_tensor.shape, mask_tensor.dtype)
    
    # Convert to NumPy arrays
    img = img_tensor.numpy()
    mask = mask_tensor.numpy()
    
    # Apply transformations
    transformed = train_transforms(image=img, mask=mask)
    dataset = [(transformed["image"], transformed["mask"]) for _ in range(4)]
    loader = DataLoader(dataset, batch_size=4)
    data, targets = next(iter(loader))
    print("Data:", data.shape, data.dtype)
    print("Targets:", targets.shape, targets.dtype)

def test_1(loader):
    """!
    @brief Tests a DataLoader by printing the shape and type of a single batch.

    @param loader (torch.utils.data.DataLoader): DataLoader to test.
    """
    data, targets = next(iter(loader))
    print("Data:", data.shape, data.dtype)
    print("Targets:", targets.shape, targets.dtype)

def test_2():
    """!
    @brief Tests the LaneDataset with a custom transformation pipeline.

    Creates a dataset with specific transformations, loads it into a DataLoader,
    and prints the shapes of images and masks for each batch.
    """
    # Define transformations with fixed size
    transform = A.Compose([
        A.Resize(height=160, width=240),  # Fixed size
        A.Rotate(limit=10),  # Random rotation
        ToTensorV2()  # Convert to tensor
    ])
    
    dataset = LaneDataset(
        image_dir="data/train_little",
        mask_dir="data/train_masks_little",
        transform=transform,
        num_augmentations=2,
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    for batch_idx, (images, masks) in enumerate(loader):
        print(f"Batch {batch_idx}: Imagens {images.shape}, Máscaras {masks.shape}")

if __name__ == "__main__":
    test()
    