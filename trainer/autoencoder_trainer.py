import sys
import os
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from cadlib.macro import (
        EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX, 
        N_ARGS_EXT, N_ARGS_PLANE, N_ARGS_TRANS, 
        N_ARGS_EXT_PARAM
        )


class TrainerEncoderDecoder:
    def __init__(self, model, loss_fn, optimizer, config, scheduler=None):
        self.model = model  # CAD transformer 
        self.loss_fn = loss_fn  # CAD loss function
        self.optimizer = optimizer   
        self.lr = config.lr 
        self.device = torch.device(f"cuda:{config.device}")
        self.grad_clip = config.grad_clip
        self.val_every = config.val_every
        self.save_every = config.save_every
        self.num_epoch = config.num_epochs
        self.log_dir = config.log_dir
        self.model_path = config.model_dir

        self.scheduler = scheduler

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

        self.model.to(self.device)

        self.model.train()

        commands = data["command"].to(self.device)
        args = data["args"].to(self.device)

        outputs = self.model(commands, args)

        loss_dict = self.loss_fn(outputs)
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
       
        self.optimizer.step()

        return outputs, loss_dict


    def validate_one_step(self, data):
        self.model.eval() 
        commands = data['command'].to(self.device)
        args = data['args'].to(self.device)

        with torch.no_grad():
            outputs = self.model(commands, args)

        loss_dict = self.loss_fn(outputs)

        return outputs, loss_dict
    

    def eval_one_epoch(self, val_loader):
        self.model.eval()

        pbar = tqdm(val_loader, leave=False)
        pbar.set_description("EVALUATE[{}]".format(self.epoch))

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

            args_comp = (gt_args == out_args).astype(int)
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
                                global_step=self.epoch)

    def _load_ckpt(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise ValueError("Checkpoint {} not exists.".format(ckpt_path))
         
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # load model 
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("\n[INFO] CSR Checkpoint successfully loaded")
        print(f"       Path: {ckpt_path}\n")


    def _save_ckpt(self):
        model_state_dict = self.model.state_dict()
        save_path = os.path.join(self.model_path, "ckpt_epoch{}.pth".format(self.epoch))
        torch.save({
            'model_state_dict': model_state_dict, 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()}, 
            save_path
        )

    def train(self, train_loader, val_loader, val_loader_all, ckpt=None):

        if ckpt is not None:
            self._load_ckpt(ckpt)

        for epoch in range(self.num_epoch):
            self.epoch += 1
            pbar = tqdm(train_loader, leave=False)

            for b, data in enumerate(pbar):
                # train one epoch 
                outputs, loss_dict = self.train_one_step(data)

                # udpate tensorboard
                if self.step % 10 == 0:
                    self._record_loss(loss_dict, 'train')

                # update pbar
                pbar.set_description("EPOCH[{}/{}]".format(epoch + 1, self.num_epoch, b))
                pbar.set_postfix(OrderedDict({k: v.item() for k, v in loss_dict.items()}))
               
                self.step += 1

                # validate one step
                if self.step % self.val_every == 0: 
                    val_data = next(val_loader)
                    outputs, loss_dict  = self.validate_one_step(val_data)
                    self._record_loss(loss_dict, mode="validation")

                self._update_scheduler(epoch)

            # validation
            if self.epoch % 5 == 0: 
                self.eval_one_epoch(val_loader_all)

            # save model: checkpoint 
            if self.epoch % self.save_every == 0: 
                pbar.set_description("saving model at: {}".format(self.model_path))
                self._save_ckpt()


        # save the final model 
        self._save_ckpt()
    
