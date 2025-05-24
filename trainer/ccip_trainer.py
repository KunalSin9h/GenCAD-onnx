import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import numpy as np
import time
import copy
import os
from PIL import Image

from collections import OrderedDict
from tqdm import tqdm
import argparse
from utils.model_utils import AvgMeter

from torch.utils.tensorboard import SummaryWriter
import einops


class TrainerCCIPModel: 
    def __init__(self, model, config, optimizer, scheduler=None): 
        self.model = model
        self.optimizer = optimizer 
        self.scheduler = scheduler
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.lr = config.lr
        self.device = config.device
        self.val_every = config.val_every
        self.log_dir = config.log_dir
        self.model_path = config.model_dir
        self.save_every = config.save_every

        self.loss_meter = AvgMeter()
        self.epoch, self.step = 0, 0

        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    def train_one_step(self, data):
        # train for one epoch 
        self.model.train() 

        batch_cmd = data["command"].to(self.device) # (B, 60)
        batch_args = data["args"].to(self.device)  # (B, 60, 16)
        batch_image = data["image"].to(self.device)  # (B, 256, 256, 3)

        # contrastive loss
        batch_cad = (batch_cmd, batch_args)
        loss = self.model(batch_cad, batch_image, return_loss=True, freeze_cad_encoder=True)

        self.optimizer.zero_grad()
        loss.backward()
       
        self.optimizer.step()

        return loss

    def val_one_epoch(self, val_loader):
        self.model.eval()

        pbar = tqdm(val_loader, leave=False)
        pbar.set_description("EVALUATE[{}]".format(self.epoch))

        for batch in pbar: 
            batch_cmd = batch["command"].to(self.device) # (B, 60)
            batch_args = batch["args"].to(self.device)  # (B, 60, 16)
            batch_image = batch["image"].to(self.device)  # (B, 256, 256, 3)

            batch_cad = (batch_cmd, batch_args)
            loss = self.model(batch_cad, batch_image, return_loss=True, freeze_cad_encoder=True)

            pbar.set_postfix(OrderedDict({"val loss": loss.item()}))

        return loss

    def _save_ckpt(self, multi_gpu=False, only_image_encoder=False):
        if only_image_encoder:
            # save only the image encoder 
            if multi_gpu:
                model_state_dict = self.model.image_encoder.state_dict()
            else: 
                model_state_dict = self.model.module.image_encoder.state_dict()

            save_path = os.path.join(self.model_path, "img_encoder_ckpt_epoch{}.pth".format(self.epoch))
            torch.save({
                'model_state_dict': model_state_dict, 
                'optimizer_state_dict': self.optimizer.state_dict()
                }, 
                save_path)
        else:
            # save the entire ccip model
            if multi_gpu:
                model_state_dict = self.model.module.state_dict()
            else: 
                model_state_dict = self.model.state_dict()


            save_path = os.path.join(self.model_path, "ckpt_epoch{}.pth".format(self.epoch))
            torch.save({
                'model_state_dict': model_state_dict, 
                'optimizer_state_dict': self.optimizer.state_dict()
                }, 
                save_path
            )

    def _load_ckpt(self, ckpt_path): 
        """
        load checkpoint for the model
        """
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print("\n[INFO] CCIP Checkpoint successfully loaded")
        print(f"       Path: {ckpt_path}\n")

        print("[INFO] Optimizer loaded\n")


    def _record_loss(self, loss, mode="train"):
        # update loss in train or validation tensor board
        losses_values = loss.item()        
        tb = self.train_tb if mode == 'train' else self.val_tb
        tb.add_scalar('training loss', losses_values, self.step)

    def train(self, train_loader, val_loader, multi_gpu=False, ckpt=None):

        if ckpt is not None:
            self._load_ckpt(ckpt)

        for epoch in range(self.num_epochs):
            self.epoch += 1
            total_loss = 0.0

            pbar = tqdm(train_loader, leave=False)

            for b, data in enumerate(pbar):
                train_loss = self.train_one_step(data) 
                pbar.set_description("EPOCH[{}/{}]".format(epoch, self.num_epochs, b))
                pbar.set_postfix(OrderedDict({"train loss": train_loss.item()}))
                if self.step % 10 == 0:
                    self._record_loss(train_loss, mode='train')

                self.step += 1
                total_loss += train_loss.item()

            if self.scheduler is not None:
                self.scheduler.step(total_loss)

            if self.epoch % self.val_every == 0: 
                val_loss = self.val_one_epoch(val_loader)
                pbar.set_description("validation")
                pbar.set_postfix(OrderedDict({"validation loss": val_loss.item()}))
                
                self._record_loss(val_loss, mode='validation')
                
            if self.epoch % self.save_every == 0: 
                pbar.set_description("saving model at: {}".format(self.model_path))
                self._save_ckpt(multi_gpu=multi_gpu)

        self._save_ckpt(multi_gpu=multi_gpu)            
