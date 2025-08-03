import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import torch.nn.functional as F
from dataset import train_transforms, val_transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
import logging
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_predictions_as_imgs_1,
    calculate_alpha,
)

# Hyperparameters
LEARNING_RATE = 0.001  #!< Learning rate for the optimizer.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  #!< Device for computation (GPU or CPU).
BATCH_SIZE = 16  #!< Number of samples per batch.
NUM_EPOCHS = 3  #!< Number of training epochs.
NUM_WORKERS = 2  #!< Number of subprocesses for data loading.
IMAGE_HEIGHT = 144  #!< Height of input images (originally 1280).
IMAGE_WIDTH = 256  #!< Width of input images (originally 1918).
PIN_MEMORY = True  #!< Whether to pin memory for faster data transfer to GPU.
LOAD_MODEL = False  #!< Whether to load a pre-trained model.
TRAIN_IMG_DIR = "data/train_little/"  #!< Directory for training images.
TRAIN_MASK_DIR = "data/train_masks_little/"  #!< Directory for training masks.
VAL_IMG_DIR = "data/val_little/"  #!< Directory for validation images.
VAL_MASK_DIR = "data/val_masks_little/"  #!< Directory for validation masks.

# Configure logging to save metrics to a file
logging.basicConfig(filename='training_log.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())  # Also display logs in terminal

def train_fn(loader, model, optimizer, loss_fn_focal, loss_fn_dice, scaler=None, epoch=0):
    """!
    @brief Trains the model for one epoch.

    This function performs a single training epoch, computing losses using Focal and Dice loss functions,
    and updates the model parameters.

    @param loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    @param model (torch.nn.Module): The model to train.
    @param optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    @param loss_fn_focal (nn.Module): Focal loss function for imbalanced classes.
    @param loss_fn_dice (nn.Module): Dice loss function for segmentation.
    @param scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision training (default: None).
    @param epoch (int, optional): Current epoch number for logging (default: 0).

    @return float: Mean training loss for the epoch.
    """
    model.train()  # Activate training mode
    loop = tqdm(loader)
    total_loss = 0
    total_batches = len(loader)
    
    optimizer.zero_grad() #alteração zerar gradientes antes de detecção
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE).float()
        
        # Forward pass with mixed precision (optional)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss_focal = loss_fn_focal(predictions, targets)
                loss_dice = loss_fn_dice(predictions, targets)
                loss = 0.7 * loss_focal + 1.3 * loss_dice
        else:
            predictions = model(data)
            loss_focal = loss_fn_focal(predictions, targets)
            loss_dice = loss_fn_dice(predictions, targets)
            loss = 0.7 * loss_focal + 1.3 * loss_dice
        
        # Backward pass
        # optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (batch_idx + 1))  # Running average loss per batch
    
    mean_loss = total_loss / total_batches
    logging.info(f"Epoch {epoch} - Mean training loss: {mean_loss:.4f}")
    return mean_loss

class FocalLoss(nn.Module):
    """!
    @brief Focal Loss for handling class imbalance in segmentation tasks.

    This class implements Focal Loss, which focuses training on hard examples by reducing the weight
    of easily classified samples.

    @param alpha (float, optional): Weight for the positive class (default: 0.95).
    @param gamma (float, optional): Focusing parameter (default: 2.0).
    @param reduction (str, optional): Reduction method for the loss ('mean', 'sum', or 'none') (default: 'mean').
    """
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for positive class
        self.gamma = gamma  # Focusing factor
        self.reduction = reduction
        print(f"FocalLoss initialized with alpha={self.alpha}, gamma={self.gamma}. "
              f"Ensure alpha reflects dataset imbalance.")

    def forward(self, inputs, targets):
        """!
        @brief Computes the Focal Loss between predictions and targets.

        @param inputs (torch.Tensor): Raw model outputs (logits).
        @param targets (torch.Tensor): Ground truth binary masks.

        @return torch.Tensor: Computed Focal Loss.
        """
        # Check for invalid values
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            logging.warning("Inputs contain NaN or Inf.")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            logging.warning("Targets contain NaN or Inf.")
        
        # Clamp logits to avoid overflow
        inputs = torch.clamp(inputs, min=-100, max=100)
        
        # Compute BCE loss with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Adjusted probabilities
        p_t = torch.sigmoid(inputs)
        p_t = targets * p_t + (1 - targets) * (1 - p_t)  # Corrected probability
        
        # Focusing factor: reduce weight of easy examples
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha-based weighting
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Final loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """!
    @brief Dice Loss for segmentation tasks.

    This class implements Dice Loss, which measures the overlap between predicted and ground truth masks.

    @note Adds a small smooth factor to avoid division by zero.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        """!
        @brief Computes the Dice Loss between predictions and targets.

        @param inputs (torch.Tensor): Raw model outputs (logits).
        @param targets (torch.Tensor): Ground truth binary masks.
        @param smooth (float, optional): Smoothing factor to avoid division by zero (default: 1e-8).

        @return torch.Tensor: Computed Dice Loss.
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

def main():
    """!
    @brief Main training loop for the U-Net model.

    This function initializes the model, loss functions, optimizer, and data loaders,
    then trains the model for the specified number of epochs, saving checkpoints and predictions.
    """
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn_focal = FocalLoss(alpha=0.95, gamma=2.0)
    loss_fn_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load data
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    best_dice_score = 0.0
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("model.pth.tar"), model)
        print("Modelo carregado com sucesso!")
        dice_score_saved = check_accuracy(val_loader, model, device=DEVICE)
        best_dice_score = dice_score_saved

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None  # Mixed precision
    
    print(f'Dice score: {best_dice_score}')
    
    for epoch in range(NUM_EPOCHS):
        # Training
        train_fn(train_loader, model, optimizer, loss_fn_focal, loss_fn_dice, scaler, epoch)
        
        # Validation
        dice_score = check_accuracy(val_loader, model, device=DEVICE)
        
        print(f"Epoch {epoch}: Dice Score = {dice_score:.4f}")
        
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename="model.pth.tar")
            print("Modelo salvo (melhor validação)!")
        
        # Save predictions as images
        save_predictions_as_imgs_1(val_loader, model, epoch, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
    main()