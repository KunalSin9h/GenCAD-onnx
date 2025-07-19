import torch
import torch.nn as nn
from model import GaussianDiffusion1D, ResNetDiffusion, VanillaCADTransformer, CLIP, ResNetImageEncoder, GenCADClipAdapter
from config import ConfigAE

class GenCADSuperModel(nn.Module):
    def __init__(self, clip_adapter, diffusion, cad_decoder):
        super().__init__()
        self.clip_adapter = clip_adapter
        self.diffusion = diffusion
        self.cad_decoder = cad_decoder

    def forward(self, img):
        # img: [B, 3, 256, 256]
        image_embed = self.clip_adapter.embed_image(img, normalization=False)
        if isinstance(image_embed, (tuple, list)):  # depending on return type
            image_embed = image_embed[0]
        latent = self.diffusion.sample(cond=image_embed)
        latent = latent.unsqueeze(1)  # [B, 256] -> [B, 1, 256]
        outputs = self.cad_decoder(None, None, z=latent, return_tgt=False)
        return outputs

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cad_ckpt_path = "model/ckpt/ae_ckpt_epoch1000.pth"
    clip_ckpt_path = "model/ckpt/ccip_sketch_ckpt_epoch300.pth"
    diffusion_ckpt_path = 'model/ckpt/sketch_cond_diffusion_ckpt_epoch1000000.pt'

    resnet_params = {"d_in": 256, "n_blocks": 10, "d_main": 2048, "d_hidden": 2048, 
                     "dropout_first": 0.1, "dropout_second": 0.1, "d_out": 256}

    # Diffusion prior
    diff_model = ResNetDiffusion(**resnet_params)
    diffusion = GaussianDiffusion1D(
        diff_model,
        z_dim=256,
        timesteps=500,
        objective='pred_x0',
        auto_normalize=False
    )
    ckpt = torch.load(diffusion_ckpt_path, map_location="cpu")
    diffusion.load_state_dict(ckpt['model'])
    diffusion = diffusion.to(device)
    diffusion.eval()

    # CCIP/CLIP Model
    cfg_cad = ConfigAE(phase="test", device=device, overwrite=False)
    cad_encoder = VanillaCADTransformer(cfg_cad)
    vision_network = "resnet-18"
    image_encoder = ResNetImageEncoder(network=vision_network)
    clip = CLIP(image_encoder=image_encoder, cad_encoder=cad_encoder, dim_latent=256)
    clip_checkpoint = torch.load(clip_ckpt_path, map_location='cpu')
    clip.load_state_dict(clip_checkpoint['model_state_dict'])
    clip.eval()
    clip_adapter = GenCADClipAdapter(clip=clip).to(device)

    # CAD Decoder
    config = ConfigAE(exp_name="test", phase="test", batch_size=1, device=device, overwrite=False)
    cad_decoder = VanillaCADTransformer(config).to(device)
    cad_ckpt = torch.load(cad_ckpt_path)
    cad_decoder.load_state_dict(cad_ckpt['model_state_dict'])
    cad_decoder.eval()

    # Combine into pipeline
    super_model = GenCADSuperModel(clip_adapter, diffusion, cad_decoder).to(device)
    super_model.eval()

    # Correct dummy input shape!
    dummy_img = torch.randn(1, 3, 256, 256, device=device)
    torch.onnx.export(
        super_model,
        dummy_img,
        "gencad_supermodel.onnx",
        input_names=['image'],
        output_names=['cad_logits'],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={
            'image': {0: 'batch_size'},
            'cad_logits': {0: 'batch_size'}
        }
    )
    print("ONNX model exported as gencad_supermodel.onnx")
