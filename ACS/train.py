import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import time
from model import UNET
import numpy as np

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODLE = True
TRAIN_IMG_DIR = "/content/drive/Shareddrives/pix2pixHD/model_crack_detection/data_set/dataset_train/train_images"
TRAIN_MASK_DIR = "/content/drive/Shareddrives/pix2pixHD/model_crack_detection/data_set/dataset_train/train_mask"
VAL_IMG_DIR = "/content/drive/Shareddrives/pix2pixHD/model_crack_detection/data_set/dataset_train/val_images"
VAL_MASK_DIR = "/content/drive/Shareddrives/pix2pixHD/model_crack_detection/data_set/dataset_train/val_mask"
MODEL = UNET(in_channels=3, out_channels=1).to(DEVICE)
 
def train_fn(loader, model, optimizer,  scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = Exponential_function_loss(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def Power_function_loss(my_preds, gt_preds):

  # q(α)*yj*log pj + (1 − yj )*log(1 − pj )
  #q(α) = beta ∗(alpha/(1-alpha))^gama

  assert my_preds.shape == gt_preds.shape
  batch_size = my_preds.shape[0]
  IMG_SIZE = my_preds.shape[1]

  eps = 0.00001
  alpha = sum(my_preds == 0.0)/(batch_size * IMG_SIZE * IMG_SIZE)
  my_preds[my_preds == 0.0] = eps
  my_preds[my_preds == 1.0] = 1.0 - eps
  q_alpha=beta*((alpha/(1-alpha))**gama)
  loss = -q_alpha*gt_preds*np.log(my_preds)
  loss -= (1-gt_preds)*np.log(1-my_preds)
  loss=loss.sum()
  return loss
  # loss = - (gt_preds * np.log(my_preds))
  # loss -= (1. - gt_preds) * np.log(1. - my_preds)
  # loss = loss.sum() / (batch_size * IMG_SIZE * IMG_SIZE)
  # return loss
def  Logarithmic_function_loss(my_preds, gt_preds):
  # loss = q(α)*yj*log pj + (1 − yj )*log(1 − pj )
  #q(α) = beta ∗ln(alpha/(1-alpha))
  assert my_preds.shape == gt_preds.shape
  batch_size = my_preds.shape[0]
  IMG_SIZE = my_preds.shape[1]

  eps = 0.00001
  alpha = len(my_preds == 0.0)/(batch_size * IMG_SIZE * IMG_SIZE)
  my_preds[my_preds == 0.0] = eps
  my_preds[my_preds == 1.0] = 1.0 - eps

  q_alpha=beta*(np.log(alpha/(1-alpha)))

  loss = -q_alpha*gt_preds*np.log(my_preds)
  loss -= (1-gt_preds)*np.log(1-my_preds)
  loss=loss.sum()
  return loss
  # loss = - (gt_preds * np.log(my_preds))
  # loss -= (1. - gt_preds) * np.log(1. - my_preds)
  # loss = loss.sum() / (batch_size * IMG_SIZE * IMG_SIZE)
  # return loss

def  Exponential_function_loss(my_preds, gt_preds):
  # loss = q(α)*yj*log pj + (1 − yj )*log(1 − pj )
  #q(α) = beta ∗ln(alpha/(1-alpha))
  
#q(α) = β ∗ a^gama*(2α−1)
  assert my_preds.shape == gt_preds.shape
  batch_size = my_preds.shape[0]
  IMG_SIZE = my_preds.shape[1]
  #
  beta=0.75
  gama=1
  a=10
  b=1
  #
  eps = 0.00001

  my_preds[my_preds == 0.0] = eps
  my_preds[my_preds == 1.0] = 1.0 - eps
  #
  alpha = len(my_preds == 0.0)/(batch_size * IMG_SIZE * IMG_SIZE)
  q_alpha=beta*a**(gama*(2*alpha-1))

  loss = -q_alpha*gt_preds*np.log(my_preds)
  loss -= (1-gt_preds)*np.log(1-my_preds)
  loss=loss.sum()
  return loss
  # loss = - (gt_preds * np.log(my_preds))
  # loss -= (1. - gt_preds) * np.log(1. - my_preds)
  # loss = loss.sum() / (batch_size * IMG_SIZE * IMG_SIZE)
  # return loss
def  Holist_function_loss(my_preds, gt_preds):
  # loss = q(α)*yj*log pj + (1 − yj )*log(1 − pj )
  #q(α) = beta ∗ln(alpha/(1-alpha))
  
#q(α) = β ∗ a^gama*(2α−1)
  assert my_preds.shape == gt_preds.shape
  batch_size = my_preds.shape[0]
  IMG_SIZE = my_preds.shape[1]

  eps = 0.00001
  
  my_preds[my_preds == 0.0] = eps
  my_preds[my_preds == 1.0] = 1.0 - eps

  #q_alpha=beta*a**(gama*(2*alpha-1))
  loss1= -alpha*gt_preds*np.log(my_preds)
  loss1-= (1-alpha)*(1-gt_preds)*np.log(1-my_preds)
  loss1=loss1.sum()
  loss2= (sum(gt_preds*my_preds) +lamda)/(sum(gt_preds)+sum(my_preds)-sum(gt_preds*my_preds)+lamda)
  loss = a*loss1 + b*(1-loss2)
  # loss = -q_alpha*gt_preds*np.log(my_preds)
  # loss -= (1-gt_preds)*np.log(1-my_preds)
  #loss=loss.sum()
  
  return loss
  # loss = - (gt_preds * np.log(my_preds))
  # loss -= (1. - gt_preds) * np.log(1. - my_preds)
  # loss = loss.sum() / (batch_size * IMG_SIZE * IMG_SIZE)
  # return loss

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_transform = val_transform

    model = MODEL
    #loss_fn = Exponential_function_loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(
            torch.load("/content/drive/MyDrive/Unet/my_checkpoint.pth.tar"),
            model,
            optimizer,
        )
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        start = time.time()
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print("time :", (time.time() - start))

        # save model
        if SAVE_MODLE:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader,
            model,
            folder="/content/drive/Shareddrives/pix2pixHD/model_crack_detection/ACS/saved_images/",
            device=DEVICE,
        )


if __name__ == "__main__":
    main()