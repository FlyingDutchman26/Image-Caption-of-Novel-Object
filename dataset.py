import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


class ImageCaptionDataset(Dataset):
    def __init__(self,mode = 'train', processor = AutoProcessor.from_pretrained("microsoft/git-base")):
        self.mode = mode
        if self.mode not in ['train','val','test']:
            raise NameError
        if self.mode == 'train':
            self.anno_path = "coco2014/annotations_DCC/captions_no_caption_rm_eightCluster_train2014.json"
            self.img_path = "coco2014/train2014"
        elif self.mode == 'val':
            self.anno_path = 'coco2014/annotations_DCC/captions_val_val2014.json'
            self.img_path = "coco2014/val2014"
        elif self.mode == 'test':
            self.anno_path = 'coco2014/annotations_DCC/captions_val_test2014.json'
            self.img_path = "coco2014/val2014"
            
        self.coco = COCO(annotation_file=self.anno_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.processor = processor
    
    def __len__(self):
        if self.mode == 'val':
            return 1000 # 让验证更迅速，选一部分
        else:
            return len(self.ids)

    
    def __getitem__(self, idx):
        img_id = self.ids[idx] # 将idx转换为img_id
        ann_ids = self.coco.getAnnIds(img_id)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        # print(path)
        img = Image.open(os.path.join(self.img_path, path)).convert('RGB')
        targets = self.coco.loadAnns(ann_ids)
        captions = []
        for target in targets:
            captions.append(target['caption'])
            encoding = self.processor(images=img, text=captions[random.randint(0,len(captions)-1)], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding['img_id'] = img_id
        return encoding