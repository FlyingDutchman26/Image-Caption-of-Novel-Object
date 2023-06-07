import os
import random

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import seed_everything
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from peft import LoraConfig, get_peft_model
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from dataset import ImageCaptionDataset
from model import LitModule

MODEL_PATH = "microsoft/git-base" # "microsoft/git-large"
TRAIN_BATCH_SIZE = 24
VAL_BATCH_SIZE = 250
LR = 5e-5
LOG_FREQ = 4
SEED = 3407
seed_everything(SEED)
torch.set_float32_matmul_precision("high")

CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value",'key', 'q_proj','v_proj','k_proj'],
    lora_dropout=0.1,
    bias="none",
    # modules_to_save=["output"], 这样做总是把所有带output的层都冻结
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = LitModule(
    model_path=MODEL_PATH,
    lora_config=CONFIG,
    # lora_config = None,
    lr=LR,
    processor=processor
)


train_set = ImageCaptionDataset(mode = 'train',processor=processor)

train_loader = DataLoader(
    train_set,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=8,
    drop_last=True,
)

val_set = ImageCaptionDataset(mode = 'val', processor=processor)

val_loader = DataLoader(
    val_set,
    batch_size=VAL_BATCH_SIZE,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=8,
    shuffle=True,
)

wandb_logger = WandbLogger(project="nndl project3")
# wandb_logger.watch(model, log_freq=LOG_FREQ, log="all")

early_stop_callback = EarlyStopping(
    monitor="val/meteor", min_delta=0, patience=5, mode="max"
)

trainer = pl.Trainer(
    precision='16-mixed',
    max_epochs=5,
    # min_epochs=10,
    accelerator="cuda",
    devices=[2],
    # strategy='ddp',
    logger=wandb_logger,
    log_every_n_steps=LOG_FREQ,
    benchmark=True,
    callbacks=[early_stop_callback],
    # accumulate_grad_batches=2,  
    # check_val_every_n_epoch=1,
    val_check_interval = 50, 
    num_sanity_val_steps=0,
    # num_sanity_val_steps=100, # 默认好像是2
    # gradient_clip_val=0.5,  # 默认是按范数裁剪
)

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

