import os
import random
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

# Assuming unet.py is in the same directory
from unet import UNet
from model.eval_metrics import soft_csi_loss, hard_csi
from model.eval_metrics import compute_fss, exp_weighted_temporal_fss

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "data_dir": "./data",      # Path to your NPZ files
    "train_split": 0.7,        # 70% training, 30% validation
    "batch_size": 4,
    "num_workers": 4,
    "max_epochs": 50,
    "devices": 1,              # Number of GPUs
    "lr": 1e-4,
    "weight_decay": 1e-5,      # Added for better regularization
    "seed": 42                 # For reproducible shuffling
}

# ==========================================
# DATASET
# ==========================================
class NPZDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Using context manager for safe file loading
        with np.load(self.files[idx]) as data:
            arr = data[data.files[0]]         # Shape: (24, 3, 112, 112)
        
        arr = np.nan_to_num(arr, nan=0.0) 

        # x: First 4 time steps (4*3 = 12 channels)
        # y: Next 6 time steps (6*3 = 18 channels)
        x = torch.tensor(arr[:4],   dtype=torch.float32).reshape(12, 112, 112)
        y = torch.tensor(arr[4:10], dtype=torch.float32).reshape(18, 112, 112)

        return x, y

# ==========================================
# LIGHTNING MODULE
# ==========================================
class UNetLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=0.0, threshold=1.0):
        super().__init__()
        self.save_hyperparameters()

        self.model = UNet(channels_in=12, channels_out=18)
        self.threshold = threshold

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x, y = batch

        pred = self(x)

        loss = soft_csi_loss(pred, y, threshold=self.threshold)

        # ---- reshape for temporal metrics ----
        with torch.no_grad():

            B, C, H, W = pred.shape

            pred_seq = pred.view(B, 6, 3, H, W)[:, :, 0]
            y_seq    = y.view(B, 6, 3, H, W)[:, :, 0]

            fss_score = compute_fss(pred_seq, y_seq, threshold=1.0, scale=5)

            tw_fss = exp_weighted_temporal_fss(
                pred_seq,
                y_seq,
                threshold=1.0,
                scale=5
            )

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_FSS", fss_score, prog_bar=False)
        self.log("train_TWFSS", tw_fss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
    
        pred = self(x)
    
        loss = soft_csi_loss(pred, y, threshold=self.threshold)
    
        csi = hard_csi(pred, y, threshold=self.threshold)
    
        # ---- reshape temporal ----
        B, C, H, W = pred.shape
    
        pred_seq = pred.view(B, 6, 3, H, W)[:, :, 0]
        y_seq    = y.view(B, 6, 3, H, W)[:, :, 0]
    
        # ---- original FSS ----
        fss_score = compute_fss(
            pred_seq,
            y_seq,
            threshold=1.0,
            scale=5
        )
    
        # ---- custom weighted FSS ----
        tw_fss = exp_weighted_temporal_fss(
            pred_seq,
            y_seq,
            threshold=1.0,
            scale=5
        )
    
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_CSI", csi, prog_bar=True)
        self.log("val_FSS", fss_score, prog_bar=True)
        self.log("val_TWFSS", tw_fss, prog_bar=True)
    
        return loss

# ==========================================
# TRAINING EXECUTION
# ==========================================
def main():
    # 1. File Preparation
    if not os.path.exists(CONFIG["data_dir"]):
        print(f"Error: Directory '{CONFIG['data_dir']}' not found.")
        return

    all_files = sorted([
        os.path.join(CONFIG["data_dir"], f) 
        for f in os.listdir(CONFIG["data_dir"]) 
        if f.endswith('.npz')
    ])
    
    # Shuffle with a fixed seed so your validation set is consistent
    random.seed(CONFIG["seed"])
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * CONFIG["train_split"])
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Files found: {len(all_files)}")
    print(f"Training on: {len(train_files)} | Validating on: {len(val_files)}")

    # 2. DataLoaders
    train_loader = DataLoader(
        NPZDataset(train_files), 
        batch_size=CONFIG["batch_size"], 
        shuffle=True,  
        num_workers=CONFIG["num_workers"], 
        pin_memory=True
    )
    val_loader = DataLoader(
        NPZDataset(val_files), 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"], 
        pin_memory=True
    )

    # 3. Trainer Setup
    checkpoint_callback = ModelCheckpoint(
        monitor="val_CSI"
        mode="max" , 
        filename="unet-best-{epoch:02d}", 
        save_top_k=3, 
    )

    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator="auto", # Automatically uses GPU/MPS if available
        devices=CONFIG["devices"],
        callbacks=[checkpoint_callback],
    )

    # 4. Start
    model = UNetLightning(
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
    threshold=1.0
)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()