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
CKPT = 'nndl project3/cx3igum4/checkpoints/epoch=0-step=2300.ckpt'
# CKPT = "nndl project3/l8wayyn6/checkpoints/epoch=0-step=700.ckpt" # 全量微调路径
TRAIN_BATCH_SIZE = 24
VAL_BATCH_SIZE = 288
TEST_BATCH_SIZE = 288
LR = 5e-5
LOG_FREQ = 4
SEED = 3407
seed_everything(SEED)
torch.set_float32_matmul_precision("high")
test_f1 = 'bus' # 设置为False不测试，否则测试相应object的f1, 如'bottle'

CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value",'key', 'q_proj','v_proj','k_proj'],
    lora_dropout=0.1,
    bias="none",
    # modules_to_save=["output"], 这样做总是把所有带output的层都冻结
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# 这行是加载checkpoint
model = LitModule.load_from_checkpoint(CKPT,lora_config = CONFIG,processor=processor,test_f1 = test_f1)

# 这行是加载原模型
# model = LitModule(
#     model_path=MODEL_PATH,
#     # lora_config=CONFIG,
#     lora_config = None,
#     lr=LR,
#     processor=processor,
#     test_f1 = test_f1
# )

test_set = ImageCaptionDataset(mode = 'test',processor=processor)


test_loader = DataLoader(
    test_set,
    batch_size=TEST_BATCH_SIZE,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=8,
)


trainer = pl.Trainer(
    accelerator="cuda",
    devices=[5],
)

trainer.test(model,test_loader)
