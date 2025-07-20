from cog import BasePredictor, Input, Path

import os 
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import h5py
import glob

from model import GaussianDiffusion1D, ResNetDiffusion, VanillaCADTransformer, CLIP, ResNetImageEncoder, GenCADClipAdapter
from utils import process_image, logits2vec, CADImageDataset

from config import ConfigAE, ConfigCCIP, ConfigDP
from cadlib.macro import (
        EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX, 
        N_ARGS_EXT, N_ARGS_PLANE, N_ARGS_TRANS, 
        N_ARGS_EXT_PARAM, EOS_IDX, MAX_TOTAL_LEN
        )

from OCC.Extend.DataExchange import write_stl_file
from multiprocessing import cpu_count
from cadlib.visualize import vec2CADsolid

from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.AIS import AIS_ColoredShape, AIS_Shape
from OCC.Core.Graphic3d import Graphic3d_NOM_SILVER
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE

from PIL import Image


diffusion_ckpt_path = 'model/ckpt/sketch_cond_diffusion_ckpt_epoch1000000.pt'
cad_ckpt_path = "model/ckpt/ae_ckpt_epoch1000.pth"
clip_ckpt_path = "model/ckpt/ccip_sketch_ckpt_epoch300.pth"

# Start Xvfb in the background
if not os.getenv('DISPLAY'):
    proc = subprocess.Popen(['Xvfb', ':99', '-screen', '0', '2048x2048x24'])
    os.environ['DISPLAY'] = ':99'

def vec_to_CAD(cad_vec):
    try:
        out_vec = cad_vec.astype(float)
        out_shape = vec2CADsolid(out_vec)
        return out_shape
    except Exception as e:
        print('cannot create CAD')

def remove_bg(image_path):
    # Replace 'path_to_your_image.jpg' with the correct path to your image
    image = Image.open(image_path)

    # Convert image to RGBA (if not already in this mode)
    image = image.convert("RGBA")

    # Make white (and shades of white) pixels transparent
    datas = image.getdata()
    new_data = []
    for item in datas:
        if item[0] > 200 and item[1] > 200 and item[2] > 200:  # Adjust these values if necessary
            new_data.append((255, 255, 255, 0))  # Making white pixels transparent
        else:
            new_data.append(item)

    image.putdata(new_data)
    image.save(image_path, "PNG")

def save_view(shape, view_type, save_path, resolution_height=2048, resolution_width=2048, remove_background=True):
    # Initialize the offscreen renderer
    offscreen_renderer = Viewer3d()
    if view_type == "iso":
        offscreen_renderer.View_Iso()
    elif view_type == "front": 
        offscreen_renderer.View_Front()
    elif view_type == "rear": 
            offscreen_renderer.View_Rear()
    elif view_type == "left": 
            offscreen_renderer.View_Left()
    elif view_type == "right": 
            offscreen_renderer.View_Right()
    elif view_type == "top": 
            offscreen_renderer.View_Top()
    elif view_type == "bottom": 
            offscreen_renderer.View_Bottom()
    else: 
        raise Exception("please choose: top, bottom, front, rear, left, right, iso")

    # offscreen renderer
    offscreen_renderer.Create()
    offscreen_renderer.SetModeShaded()

    cornflower_blue_color = Quantity_Color(0.39, 0.58, 0.93, Quantity_TOC_RGB)
    # Display the shape
    # Graphic3d_NOM_TRANSPARENT, Graphic3d_NOM_SILVER
    offscreen_renderer.DisplayShape(shape, update=True, material=Graphic3d_NOM_SILVER)

    colored_shape = AIS_ColoredShape(shape)
    colored_shape.SetColor(cornflower_blue_color)
    offscreen_renderer.Context.Display(colored_shape, True)

    # -------------------------
    black_color = Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB)  # Black color for edges

    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()  # No need to re-wrap as TopoDS_Edge
        edge_shape = AIS_Shape(edge)
        edge_shape.SetColor(black_color)
        edge_shape.SetWidth(10.0)  # Set the edge width (adjust as needed)
        offscreen_renderer.Context.Display(edge_shape, True)
        exp.Next()

    # -------------------------

    offscreen_renderer.View.SetBackgroundColor(0, 1, 1, 1)
    offscreen_renderer.View.Redraw()


    # Fit the entire shape in the view
    offscreen_renderer.View.FitAll(0.5)

    # Set a high resolution for the renderer
    high_resolution_width = resolution_height
    high_resolution_height = resolution_width
    offscreen_renderer.SetSize(high_resolution_width, high_resolution_height)                


    # Render and save the image in high resolution
    offscreen_renderer.View.Dump(save_path)
    if remove_background:
        remove_bg(save_path)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        resnet_params = {"d_in": 256, "n_blocks": 10, "d_main": 2048, "d_hidden": 2048, 
                "dropout_first": 0.1, "dropout_second": 0.1, "d_out": 256}

        device_num = 0
        device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
        phase = "test"
        batch_size = 64

        self.device

        # Load diffusion prior model 

        model = ResNetDiffusion(d_in=resnet_params["d_in"], n_blocks=resnet_params["n_blocks"], 
                            d_main=resnet_params["d_main"], d_hidden=resnet_params["d_hidden"], 
                            dropout_first=resnet_params["dropout_first"], dropout_second=resnet_params["dropout_second"], 
                            d_out=resnet_params["d_out"])

        diffusion = GaussianDiffusion1D(
            model,
            z_dim=256,
            timesteps = 500,
            objective = 'pred_x0', 
            auto_normalize=False
        )

        ckpt = torch.load(diffusion_ckpt_path, map_location="cpu")
        diffusion.load_state_dict(ckpt['model'])
        diffusion = diffusion.to(device)

        diffusion.eval()

        self.diffusion = diffusion
        print('# # Diffusion checkpoint successfully loaded')

        # Load CCIP model 

        cfg_cad = ConfigAE(phase=phase, device=device, overwrite=False)
        cad_encoder = VanillaCADTransformer(cfg_cad)

        vision_network = "resnet-18"
        image_encoder = ResNetImageEncoder(network=vision_network)

        clip = CLIP(image_encoder=image_encoder, cad_encoder=cad_encoder, dim_latent=256)
        clip_checkpoint = torch.load(clip_ckpt_path, map_location='cpu')
        clip.load_state_dict(clip_checkpoint['model_state_dict'])

        clip.eval()

        self.clip = clip
        print('# # CCIP checkpoint successfully loaded')

        # Load CAD decoder model 
        # placeholder values for the config file 
        config = ConfigAE(exp_name="test", 
                phase="test", batch_size=1, 
                device=device, 
                overwrite=False)

        cad_decoder = VanillaCADTransformer(config).to(config.device) 

        cad_ckpt = torch.load(cad_ckpt_path, map_location=device)
        cad_decoder.load_state_dict(cad_ckpt['model_state_dict'])
        cad_decoder.eval()

        self.cad_decoder = cad_decoder
        print('# # CAD checkpoint successfully loaded')

    def predict(
        self,
        image: Path = Input(description="CAD isometric drawing image"),
        stl: bool = Input(description="Generate STL", default=False),
    ) -> Path:
        """Run a single prediction on the model"""

        img = process_image(image).to(self.device)
        image_embed = self.clip_adapter.embed_image(img, normalization = False)
        latent = self.diffusion.sample(cond=image_embed)
        latent = latent.unsqueeze(0) # (1, 256) --> (1, 1, 256

        with torch.no_grad():
            outputs = self.cad_decoder(None, None, z=latent, return_tgt=False)
            batch_out_vec = logits2vec(outputs, device=self.device)
            # begin loop vec: [4, -1, -1, ...., -1] 
            begin_loop_vec = np.full((batch_out_vec.shape[0], 1, batch_out_vec.shape[2]), -1, dtype=np.int64)
            begin_loop_vec[:, :, 0] = 4

            auto_batch_out_vec = np.concatenate([begin_loop_vec, batch_out_vec], axis=1)[:, :MAX_TOTAL_LEN, :]  # (B, 60, 17)

        out_vec = auto_batch_out_vec[0]
        out_command = out_vec[:, 0]

        seq_len = out_command.tolist().index(EOS_IDX)
        cad_vec = out_vec[:seq_len]
        shape = vec_to_CAD(cad_vec=cad_vec)

         # currently we only support either png or stl.
        if stl:
            export_path = "cad.stl"
            write_stl_file(shape, export_path, mode="binary", linear_deflection=0.5, angular_deflection=0.3,)
            return Path(export_path)
        else:
            img_export_path = "cad.png"
            save_view(shape, view_type='iso', save_path=img_export_path)
            return Path(img_export_path)
