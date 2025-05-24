import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import _get_padding_mask, _get_visibility_mask
from cadlib.macro import CMD_ARGS_MASK


class CADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK).to(cfg.device))

    def forward(self, output):
        # Target & predictions
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]

        # exclude firt token to make autoregressive
        tgt_commands, tgt_args = tgt_commands[:, 1:], tgt_args[:, 1:, :]

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        command_logits, args_logits = output["command_logits"], output["args_logits"]

        # exclude last token to make autoregressive
        command_logits, args_logits = command_logits[:, :-1], args_logits[:, :-1, :]
        
        mask = self.cmd_args_mask[tgt_commands.long()]

        c1 = command_logits[padding_mask.bool()].reshape(-1, self.n_commands)
        c2 = tgt_commands[padding_mask.bool()].reshape(-1).long()

        a1 = args_logits[mask.bool()].reshape(-1, self.args_dim)
        a2 = tgt_args[mask.bool()].reshape(-1).long() + 1

        loss_cmd = F.cross_entropy(c1, c2)
        loss_args = F.cross_entropy(a1, a2)  # shift due to -1 PAD_VAL

        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res



class CLIPLoss(nn.Module):
    def __init__(self): 
        super().__init__()
        self.loss_name = "clip_loss"

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def forward(self, logits, targets):         
        cad_loss = self.cross_entropy(logits, targets, reduction='none')
        image_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (image_loss + cad_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


class CCIPLoss(nn.Module):
    """This clip loss follows the openclip implementation from here: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
    """
    def __init__(self): 
        super().__init__()
        self.loss_name = "clip_loss_open_ai"
        self.labels = {}
        self.prev_num_logits = 0 

    def get_logits(self, image_features, cad_features, logit_scale):
        logits_per_image = logit_scale * image_features @ cad_features.T 
        logtis_per_cad = logit_scale * cad_features @ image_features.T 

        return logits_per_image, logtis_per_cad

    def get_ground_truth(self, device, num_logits):
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)

        else: 
            labels= self.labels[device]
        return labels


    def forward(self, image_features, cad_features, logit_scale, output_dict=True):         
        device = image_features.device

        logits_per_image, logtis_per_cad = self.get_logits(image_features, cad_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels) + 
            F.cross_entropy(logtis_per_cad, labels)
        ) / 2

        return {"constrastive_loss": total_loss} if output_dict else total_loss
