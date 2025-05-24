import os
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter

from model import CADTransformer
from .base import BaseTrainer
from .loss import CADLoss
from .scheduler import GradualWarmupScheduler
from cadlib.macro import (
        EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX, 
        N_ARGS_EXT, N_ARGS_PLANE, N_ARGS_TRANS, 
        N_ARGS_EXT_PARAM
        )


class TrainerEncoderDecoder:
    def __init__(self, model, loss_fn, optimizer, lr, 
                 scheduler, device, grad_clip=None, 
                 val_frequencey=None,
                 num_epoch=None, 
                 log_dir=None):
        self.model = model  # CAD transformer 
        self.loss_fn = loss_fn  # CAD loss function
        self.optimizer = optimizer   
        self.lr = lr 
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.val_frequencey = val_frequencey
        self.num_epoch = num_epoch
        self.log_dir = log_dir

        self.step, self.epoch = 0, 0

        # set up tensor board
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))


    def _update_scheduler(self, epoch): 
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], epoch)
        self.scheduler.step()

    def _record_loss(self, loss_dict, mode="train"):
        # update loss in train or validation tensor board
        losses_values = {k: v.item() for k, v in loss_dict.items()}
        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.step)


    def train_one_step(self, data): 
        # train for one step

        self.model.train()

        commands = data["command"].to(self.device)
        args = data["args"].to(self.device)

        outputs = self.model(commands, args)
        loss_dict = self.loss_fn(outputs)
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
       
        return outputs, loss_dict


    def eval_one_epoch(self, val_loader):
        self.model.eval()

        pbar = tqdm(val_loader)

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            commands = data['command'].to(self.device)
            args = data['args'].to(self.device)

            gt_commands = commands.squeeze(1).long().detach().cpu().numpy() # (N, S)
            gt_args = args.squeeze(1).long().detach().cpu().numpy() # (N, S, n_args)

            with torch.no_grad():
                outputs = self.model(commands, args)

            out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1
            out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)


            ext_pos = np.where(gt_commands == EXT_IDX)
            line_pos = np.where(gt_commands == LINE_IDX)
            arc_pos = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(np.int)
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :N_ARGS_PLANE])
        sket_trans_acc = np.mean(all_ext_args_comp[:, N_ARGS_PLANE:N_ARGS_PLANE+N_ARGS_TRANS])
        extent_one_acc = np.mean(all_ext_args_comp[:, -N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))


        self.val_tb.add_scalars("args_acc",
                                {"line": line_acc, "arc": arc_acc, "circle": circle_acc,
                                 "plane": sket_plane_acc, "trans": sket_trans_acc, "extent": extent_one_acc},
                                global_step=self.clock.epoch)


    def _save_ckpt(self):
        model_state_dict = self.model.cpu().state_dict()
        save_path = os.path.join(self., "ckpt_epoch{}.pth".format(self.clock.epoch))
        torch.save({
            'clock': self.clock.make_checkpoint(),
            'model_state_dict': model_state_dict, 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()}. 
            save_path
        )


    def _save_ckpt(self, epoch): 
        pass

    def train(self, train_loader, val_loader):
        for epoch in range(self.num_epoch):
            pbar = tqdm(train_loader)

            for b, data in enumerate(pbar):
                # train one epoch 
                outputs, loss_dict = self.train_one_step(data)

                self._update_scheduler(epoch)

                # update pbar
                pbar.set_description("EPOCH[{}][{}]".format(epoch, b))
                pbar.set_postfix(OrderedDict({k: v.item() for k, v in loss_dict.items()}))

                self.step += 1

            if epoch % self.val_frequencey == 0: 
                self.eval_one_epoch(val_loader)

            if epoch % self.save_every == 0: 
                self._save_ckpt(epoch)

            self.epoch += 1

        # save the final model 
        self._save_ckpt(epoch)


