import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

# Assuming these local modules are in your path
from earthformer.model_arch import EarthformerModel
from csi_eval import soft_csi_loss, hard_csi

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "data_dir": "./data",      # Path to your NPZ files
    "train_split": 0.8,
    "batch_size": 2,
    "num_workers": 4,
    "max_epochs": 50,
    "devices": 1,
    "lr": 1e-4,
    "threshold": 1.0,          # Rain threshold for CSI
    "seed": 42
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
        with np.load(self.files[idx]) as data:
            arr = data[data.files[0]]         # (24, 3, 112, 112)
            
        arr = np.nan_to_num(arr, nan=0.0) 

        # Earthformer expects (T, H, W, C)
        # Input: first 4 steps | Target: next 6 steps
        x = torch.tensor(arr[:4],   dtype=torch.float32).permute(0, 2, 3, 1)  
        y = torch.tensor(arr[4:10], dtype=torch.float32).permute(0, 2, 3, 1)  

        return x, y

# ==========================================
# LIGHTNING MODULE
# ==========================================
class EarthformerLightning(EarthformerModel):
    def __init__(self, lr=1e-4, weight_decay=0.0, threshold=1.0, **kwargs):
        # EarthformerModel likely handles the optimizer config internally via kwargs
        super().__init__(lr=lr, weight_decay=weight_decay, **kwargs)
        self.threshold = threshold

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        # Using soft_csi for gradients
        loss = soft_csi_loss(pred, y, threshold=self.threshold)  
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = soft_csi_loss(pred, y, threshold=self.threshold)
        # Monitoring hard CSI (the "real" metric)
        csi  = hard_csi(pred, y, threshold=self.threshold)       
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_CSI",  csi,  prog_bar=True)
        return loss

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # 1. Prepare Data Files
    if not os.path.exists(CONFIG["data_dir"]):
        print(f"Error: data_dir '{CONFIG['data_dir']}' not found.")
        return

    all_files = sorted([
        os.path.join(CONFIG["data_dir"], f) 
        for f in os.listdir(CONFIG["data_dir"]) 
        if f.endswith('.npz')
    ])
    
    random.seed(CONFIG["seed"])
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * CONFIG["train_split"])
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Dataset: {len(all_files)} files | Train: {len(train_files)} | Val: {len(val_files)}")

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

    # 3. Training setup
    # Note: mode="max" because we want the highest CSI score
    checkpoint_cb = ModelCheckpoint(
        monitor="val_CSI", 
        filename="earthformer-best-{epoch:02d}", 
        save_top_k=3, 
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator="auto",
        devices=CONFIG["devices"],
        callbacks=[checkpoint_cb],
    )

    # 4. Initialize and Run
    model = EarthformerLightning(
        lr=CONFIG["lr"], 
        threshold=CONFIG["threshold"]
    )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()