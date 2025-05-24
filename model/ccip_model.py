import math
import copy
from contextlib import contextmanager
from functools import partial, wraps

import torch
import torch.nn.functional as F
import torch.distributed as distributed
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from x_clip.mlm import MLM
from x_clip.visual_ssl import SimSiam, SimCLR
from x_clip.distributed import all_gather


from collections import namedtuple

# helper functions

def identity(t, *args, **kwargs):
    return t

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# checkpointing helper function

def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


def resize_image_to(
    image,
    target_image_size,
    clamp_range = None,
    nearest = False,
    **kwargs
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    if not nearest:
        scale_factors = target_image_size / orig_image_size
        out = resize(image, scale_factors = scale_factors, **kwargs)
    else:
        out = F.interpolate(image, target_image_size, mode = 'nearest')

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out

# image normalization functions
# ddpms expect images to be in the range of -1 to 1
# but CLIP may otherwise

def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)



# helper classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# patch dropout

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x, force_keep_all = False):
        if not self.training or self.prob == 0. or force_keep_all:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]



# contrastive learning functions

def model_forward_with_context(
    *,
    fn,
    args,
    kwargs,
    freeze,
):
    encoding_context = null_context if not freeze else torch.no_grad

    with encoding_context():
        if kwargs is not None:
            enc = fn(*args, **kwargs)
        else:
            enc = fn(*args)

        if freeze:
            enc = enc.clone()
            enc.detach_()

    return enc

# main clip class

class CLIP(nn.Module):
    def __init__(
        self,
        *,
        image_encoder = None,
        cad_encoder = None,
        dim_cad = 256,
        dim_image = 512 * 8 * 8,
        dim_latent = 256,
        extra_latent_projection = False,
        decoupled_contrastive_learning = False,
        sim_reg_loss_weight = 0.,
        **kwargs
    ):
        super().__init__()

        # store some parameters for access
        self.dim_cad = dim_cad
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        # instantiate cad transformer
        freeze_model_and_make_eval_(cad_encoder)
        self.cad_encoder = cad_encoder

        # instantiate image transformer
        self.image_encoder = image_encoder

        # cad latent projection
        # self.to_cad_latent = nn.Linear(dim_cad, dim_latent)

        self.to_cad_latent = nn.Sequential(nn.Linear(dim_cad, dim_latent), 
                                        nn.Linear(dim_latent, dim_latent), 
                                        nn.Tanh())
 
        # image latent projection
        self.to_visual_latent = nn.Sequential(nn.Linear(dim_image, dim_latent), nn.Tanh())
        
        # temperature
        self.temperature = nn.Parameter(torch.tensor(1.))

        # proposed in https://arxiv.org/abs/2110.06848 (DCL) and https://arxiv.org/abs/2110.11316 (CLOOB)
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        # proposed in https://arxiv.org/abs/2110.11316 (CLOOB)
        self.extra_latent_projection = extra_latent_projection

        self.to_cad_latent_extra = copy.deepcopy(self.to_cad_latent)
        self.to_visual_latent_extra = copy.deepcopy(self.to_visual_latent)

        # is distributed or not
        self.requires_all_gather = distributed.is_initialized() and distributed.get_world_size() > 1

        # use the similarity regularization proposed in https://arxiv.org/abs/2309.08773
        self.sim_reg_loss_weight = sim_reg_loss_weight
        self.has_sim_reg_loss = sim_reg_loss_weight > 0.

    def forward(
        self,
        cad = None,
        image = None,
        return_loss = False,
        return_encodings = False,
        return_latents = False,
        freeze_image_encoder = False,   # image encoder is not trained if this is set to True, proposed by LiT paper
        freeze_cad_encoder = False,    # cad encoder is not trained if this is set to True
        cad_to_image = True,           # in the case the extra projection is turned on, would return different similarity values depending on modality directionality
    ):
        batch, device = image.shape[0], image[0].device

        # concat augmented texts and images and do some asserts

        num_batch_cads = num_batch_images = 1

        # get encoded cad
        batch_cmd, batch_args = cad[0], cad[1]
        cad_args = (batch_cmd, batch_args)
        cad_kwargs = {'encode_mode': True}  # we only need the encoder

        with torch.no_grad():
            enc_cad = self.cad_encoder(batch_cmd, batch_args, encode_mode=True)

        enc_image = self.image_encoder(image)
        
        # early return of encodings, if needed

        if return_encodings:
            return enc_cad, enc_image

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only

        cad_embeds = enc_cad.squeeze() if enc_cad.ndim == 3 else enc_cad
        image_embeds = enc_image

        # project to latents
                
        cad_latents = self.to_cad_latent(cad_embeds)

        image_latents = self.to_visual_latent(image_embeds)

        cad_latents, image_latents = map(l2norm, (cad_latents, image_latents))

        # calculate another set of latents for image to cad (vs cad to image)
        # proposed by CLOOB

        cad_latents_extra, image_latents_extra = cad_latents, image_latents
        if self.extra_latent_projection:
            cad_latents_extra = self.to_cad_latent_extra(cad_embeds)
            image_latents_extra = self.to_visual_latent_extra(image_embeds)
            text_latents_extra, image_latents_extra = map(l2norm, (cad_latents_extra, image_latents_extra))

        # whether to early return latents

        if return_latents:
            if self.extra_latent_projection:
                return cad_latents, image_latents, text_latents_extra, image_latents_extra

            return cad_latents, image_latents

        # get temperature

        temp = self.temperature.exp()

        # early return, if needed

        if not return_loss:
            einsum_args = (cad_latents_extra, image_latents_extra) if self.extra_latent_projection and not cad_to_image else (cad_latents, image_latents)
            return einsum('b d, b d -> b', *einsum_args) * temp

        # split out multiview dimension for cad and images

        cad_latents = rearrange(cad_latents, '(m b) ... -> m b ...', m = num_batch_cads)
        image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m = num_batch_images)

        if self.extra_latent_projection:
            cad_latents_extra = rearrange(cad_latents_extra, '(m b) ... -> m b ...', m = num_batch_cads)
            image_latents_extra = rearrange(image_latents_extra, '(m b) ... -> m b ...', m = num_batch_images)

        # maybe distributed all gather

        if self.requires_all_gather:
            latents = torch.stack((cad_latents, image_latents))
            latents, sizes = all_gather(latents, 2, None)
            cad_latents, image_latents = latents

            batch = sizes.sum().item()

            if self.extra_latent_projection:
                latents_extra = torch.stack((cad_latents_extra, image_latents_extra))
                latents_extra, _ = all_gather(latents_extra, 2, sizes)
                cad_latents_extra, image_latents_extra = latents_extra

        # maybe similarity regularize

        sim_reg_loss = 0.

        if self.has_sim_reg_loss:
            diag_mask = torch.eye(batch, device = device, dtype = torch.bool)
            off_diag_mask = rearrange(~diag_mask, '... -> 1 ...')

            cad_sim, image_sim, cad_extra_sim, image_extra_sim = map(lambda t: einsum('m i ... d, m j ... d -> m ... i j', t, t)[off_diag_mask], (text_latents, image_latents, cad_latents_extra, image_latents_extra))

            sim_reg_loss = (
                F.mse_loss(cad_sim, image_sim) +
                F.mse_loss(cad_extra_sim, image_extra_sim)
            ) / 2

        # contrastive loss
        # 
        # m - num batches of text (for multiview)
        # n - num batches of images (for multiview)
        # x - batches of text
        # y - batches of images
        # t - sequence dimension along text tokens
        # i - sequence dimension along image tokens
        # 

        cad_to_image = einsum('m t d, n i d -> m n t i', cad_latents, image_latents) * temp
        image_to_cad = rearrange(cad_to_image, '... t i -> ... i t')

        if self.extra_latent_projection:
            image_to_cad = einsum('m t d, n i d -> m n i t', cad_latents_extra, image_latents_extra) * temp

        # calculate loss

        cad_to_image = rearrange(cad_to_image, 'm n ... -> (m n) ...')
        image_to_cad = rearrange(image_to_cad, 'm n ... -> (m n) ...')

        # exponentiate

        cad_to_image_exp, image_to_cad_exp = map(torch.exp, (cad_to_image, image_to_cad))

        # numerators

        cad_to_image_pos, image_to_cad_pos = map(matrix_diag, (cad_to_image_exp, image_to_cad_exp))

        # denominator

        if self.decoupled_contrastive_learning:
            pos_mask = torch.eye(batch, device = device, dtype = torch.bool)
            cad_to_image_exp, image_to_cad_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (cad_to_image_exp, image_to_cad_exp))

        cad_to_image_denom, image_to_cad_denom = map(lambda t: t.sum(dim = -1), (cad_to_image_exp, image_to_cad_exp))

        # loss

        cad_to_image_loss = (-log(cad_to_image_pos) + log(cad_to_image_denom)).mean(dim = -1)
        image_to_text_loss = (-log(image_to_cad_pos) + log(image_to_cad_denom)).mean(dim = -1)

        # calculate CL loss

        cl_losses = (cad_to_image_loss + image_to_text_loss) / 2


        # get main CL loss vs multiview CL losses

        cl_loss, multiview_cl_loss = cl_losses[0], cl_losses[1:]

        loss = cl_loss

        # add similarity regularization loss with weight if needed

        if self.has_sim_reg_loss:
            loss = loss + sim_reg_loss * self.sim_reg_loss_weight

        return loss
    



EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings'])
EmbeddedImage = namedtuple('EmbedImageReturn', ['image_embed', 'image_encodings'])

class BaseClipAdapter(nn.Module):
    def __init__(self, clip):
        super().__init__()
        freeze_model_and_make_eval_(clip)
        self.clip = clip

    @property
    def dim_latent(self):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def image_channels(self):
        raise NotImplementedError

    @property
    def max_text_len(self):
        raise NotImplementedError

    def embed_image(self, image):
        raise NotImplementedError



class XClipAdapter(BaseClipAdapter):
    @property
    def dim_latent(self):
        return self.clip.dim_latent

    @property
    def image_size(self):
        return self.clip.image_size

    @property
    def image_channels(self):
        return self.clip.image_channels

    @property
    def max_text_len(self):
        return self.clip.text_seq_len

    @torch.no_grad()
    def embed_cad(self, cad, normalization=True):
        """
        cad --> (B, d_model)
        returns: 
            cad_latents, cad_encodings
        """
        commands, args = cad
        encoder_output = self.clip.cad_encoder(commands, args, encode_mode=True)  # (B, 1, d_model)

        cad_cls, cad_encodings = encoder_output.squeeze(), encoder_output.squeeze()  # (B, d_model)
        cad_embed = self.clip.to_cad_latent(cad_cls)

        if normalization: 
            return EmbeddedText(l2norm(cad_embed), cad_encodings)
        else: 
            return EmbeddedText(cad_embed, cad_encodings)

    @torch.no_grad()
    def embed_image(self, image, normalization=True):
        """returns: 
            image_latents, image_encodings
        """
        encoder_output = self.clip.image_encoder(image)
        image_cls, image_encodings = encoder_output, encoder_output
        image_embed = self.clip.to_visual_latent(image_cls)

        if normalization: 
            return EmbeddedImage(l2norm(image_embed), image_encodings)
        else: 
            return EmbeddedImage(image_embed, image_encodings)



class GenCADClipAdapter(BaseClipAdapter):
    @property
    def dim_latent(self):
        return self.clip.dim_latent

    @property
    def image_size(self):
        return self.clip.image_size

    @property
    def image_channels(self):
        return self.clip.image_channels

    @property
    def max_text_len(self):
        return self.clip.text_seq_len

    @torch.no_grad()
    def embed_cad(self, cad, normalization=True):
        """
        cad --> (B, d_model)0
        returns: 
            cad_latents, cad_encodings
        """
        commands, args = cad
        # encoder_output = self.clip.cad_encoder(commands, args, encode_mode=True)  # (B, 1, d_model)

        # with torch.no_grad():
        encoder_output = self.clip.cad_encoder(commands, args, encode_mode=True)

        cad_embed = encoder_output.squeeze()

        if normalization: 
            return l2norm(cad_embed)
        else: 
            return cad_embed


    def embed_image(self, image, normalization=True):
        """returns: 
            image_latents, image_encodings
        """
        with torch.no_grad():
            encoder_output = self.clip.image_encoder(image)
        image_encodings = encoder_output
    
        with torch.no_grad():
            image_embed = self.clip.to_visual_latent(image_encodings)

        if normalization: 
            return l2norm(image_embed)
        else: 
            return image_embed
