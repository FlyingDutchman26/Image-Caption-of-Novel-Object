import collections
import json
import os
import random
from math import ceil, fabs
from typing import Any, Optional

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# from aac_metrics.functional import meteor, spice
# from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents
from lightning.pytorch.utilities.types import STEP_OUTPUT
from peft import LoraConfig, get_peft_model
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from pycocoevalcap.eval import COCOEvalCap
from utils import print_trainable_parameters

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value",'q_proj','v_proj'],
    lora_dropout=0.1,
    bias="none",
    # modules_to_save=["output"], 这样做总是把所有output层都解冻
)


class LitModule(pl.LightningModule):
    def __init__(self, model_path = "microsoft/git-base",lora_config = None, lr = 5e-5,processor = None, test_f1 = False):
        super(LitModule, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = lora_config
        self.processor = processor
        if self.config:
            self.model = get_peft_model(self.model, config)
        for name, param in  self.model.named_parameters():
            # print(name)
            if name.startswith('base_model.model.output.'): # 本来start with output即可，这里需要改
                param.requires_grad = True
        print_trainable_parameters(self.model)
        # 训练超参数
        self.lr = lr
        self.val_annotation_file = 'coco2014/annotations_DCC/captions_val_val2014.json'
        self.val_coco = COCO(self.val_annotation_file)
        self.test_f1 = test_f1 # 还可以设置为object名字比如'bottle'
        if test_f1 == False:
            self.test_annotation_file = 'coco2014/annotations_DCC/captions_val_test2014.json'
        else:
            # 将novel object换成自己所需
            self.test_annotation_file = f'coco2014/annotations_DCC/captions_split_set_{test_f1}_val_test_novel2014.json'
        self.test_coco = COCO(self.test_annotation_file)
        
        
                
    def forward(self,input_ids,attention_mask,pixel_values,labels):
        # print('debug:',pixel_values.shape)
        # assert 0 
        outputs = self.model(input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                labels = labels)
        return outputs
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}


    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["input_ids"])
        loss = outputs.loss

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # 让batch_size尽量大一些
        img_ids = batch['img_id'] #[b]
        pixel_values = batch['pixel_values'] # [b,3,224,224]
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        res = []
        img_ids = batch['img_id']
        for i in range(len(img_ids)):
            res.append({"image_id":img_ids[i].item(),'caption':generated_caption[i]})
            # res[img_ids[i].item()] = generated_caption[i]
        with open('batch_predict.json','w+') as f:
            json.dump(res,f)
        
        results_file = 'batch_predict.json'

        # create coco object and coco_result object
        coco_result = self.val_coco.loadRes(results_file)

        # create coco_eval object by taking coco and coco_result
        coco_eval = COCOEvalCap(self.val_coco, coco_result)

        # evaluate on a subset of images by setting
        # coco_eval.params['image_id'] = coco_result.getImgIds()
        # please remove this line when evaluating the full validation set
        coco_eval.params['image_id'] = coco_result.getImgIds()

        # evaluate results
        # SPICE will take a few minutes the first time, but speeds up due to caching
        coco_eval.evaluate()

        # print output evaluation scores
        result = {}
        for metric, score in coco_eval.eval.items():
            result[metric] = score
        spice = result['SPICE']
        meteor = result['METEOR']
        cider = result['CIDEr']
        self.log(
            'val/spice',
            spice,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        self.log(
            'val/meteor',
            meteor,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        self.log(
            'val/CIDEr',
            cider,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
            
            
    def test_step(self, batch, batch_idx):
        if self.test_f1 == False:
            # 让batch_size尽量大一些
            img_ids = batch['img_id'] #[b]
            pixel_values = batch['pixel_values'] # [b,3,224,224]
            generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            res = []
            img_ids = batch['img_id']
            for i in range(len(img_ids)):
                res.append({"image_id":img_ids[i].item(),'caption':generated_caption[i]})
                # res[img_ids[i].item()] = generated_caption[i]
            with open('batch_predict.json','w+') as f:
                json.dump(res,f)
            
            results_file = 'batch_predict.json'

            # create coco object and coco_result object
            coco_result = self.test_coco.loadRes(results_file)

            # create coco_eval object by taking coco and coco_result
            coco_eval = COCOEvalCap(self.test_coco, coco_result)

            # evaluate on a subset of images by setting
            # coco_eval.params['image_id'] = coco_result.getImgIds()
            # please remove this line when evaluating the full validation set
            coco_eval.params['image_id'] = coco_result.getImgIds()

            # evaluate results
            # SPICE will take a few minutes the first time, but speeds up due to caching
            coco_eval.evaluate()

            # print output evaluation scores
            result = {}
            for metric, score in coco_eval.eval.items():
                result[metric] = score
            spice = result['SPICE']
            meteor = result['METEOR']
            cider = result['CIDEr']
            self.log(
                'test/spice',
                spice,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            
            self.log(
                'test/meteor',
                meteor,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            
            self.log(
                'test/CIDEr',
                cider,
                # on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        else:
            keyword = self.test_f1  # 如 bottle
            img_ids = batch['img_id'] #[b]
            pixel_values = batch['pixel_values'] # [b,3,224,224]
            generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            TP,FP,FN,TN = 0,0,0,0
            for idx, img_id in enumerate(img_ids):
                pred_caption = generated_caption[idx] # 这是预测caption
                ann_ids = self.test_coco.getAnnIds(int(img_id)) # 获取对应的ann_ids
                targets = self.test_coco.loadAnns(ann_ids)
                label_captions = ''
                for target in targets:
                    label_captions += target['caption']
                # 获取到所有annotation,连成一个长字符串
                pred_caption = pred_caption.lower()
                label_captions = label_captions.lower()
                if keyword in pred_caption and keyword in label_captions:
                    TP += 1
                if keyword in pred_caption and keyword not in label_captions:
                    FP += 1
                if keyword not in pred_caption and keyword in label_captions:
                    FN += 1
                else:
                    TN += 1
            precision = TP/(TP + FP) if TP + FP else 1
            recall = TP/(TP + FN) if TP + FN else 1
            f1 = 2 * precision * recall/(precision + recall) if precision + recall else 0
            
            self.log(
                f'test/f1 of {keyword}',
                f1,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
                
                
                
                
                
    

