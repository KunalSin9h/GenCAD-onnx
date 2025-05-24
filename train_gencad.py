import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import argparse
import sys
import h5py


from model import VanillaCADTransformer, CLIP, ResNetImageEncoder, ViT, GaussianDiffusion1D, ResNetDiffusion
from config import ConfigAE, ConfigCCIP, ConfigDP
from trainer import TrainerEncoderDecoder, TrainerCCIPModel, Trainer1D
from utils import CADLoss, get_dataloader, get_ccip_dataloader, GradualWarmupScheduler, cycle, count_params, print_training_complete
from utils.cad_dataset import DPDataset


def train_model(model="autoencoder", args=None):

    if model=="csr":

        print(f"\n[INFO] GPU: {args.gpu}\n")

        # load config

        config = ConfigAE(exp_name=args.exp_name, device=args.gpu, load_ckpt=args.ckpt_path)

        # data loader: each batch contains: command, args and id

        train_loader = get_dataloader(phase="train", config=config)
        val_loader_all = get_dataloader(phase="validation", config=config)
        val_loader = get_dataloader(phase="validation", config=config)
        val_loader = cycle(val_loader)


        # CSR model 

        model = VanillaCADTransformer(config).to(config.device)        

        count_params(model)  # print total trainable parameters

        loss_fn = CADLoss(config)

        optimizer = optim.Adam(model.parameters(), config.lr)

        if config.use_scheduler: 
            scheduler = GradualWarmupScheduler(optimizer, 1.0, config.warmup_step)
        else: 
            scheduler = None

        # trainer 

        ae_trainer = TrainerEncoderDecoder(model, loss_fn, optimizer, config, scheduler)

        # train model 

        ae_trainer.train(train_loader=train_loader, val_loader=val_loader, val_loader_all=val_loader_all, ckpt=args.ckpt_path) 


    elif model=="ccip": 

        print(f"\n[INFO] GPU: {args.gpu}\n")

        # load config

        phase = "train"

        cfg_cad = ConfigAE(phase=phase, overwrite=False)  # config file for AE is needed to load the model, inputs are not important

        cad_encoder = VanillaCADTransformer(cfg_cad)  # load cad autoencoder checkpoint 
        cad_checkpoint = torch.load(args.cad_ckpt_path, map_location='cpu')
        cad_encoder.load_state_dict(cad_checkpoint['model_state_dict'])

        print("[INFO] CSR Checkpoint successfully loaded")
        print(f"       Path: {args.cad_ckpt_path}\n")

        vision_network = "resnet-18"
        image_encoder = ResNetImageEncoder(network=vision_network)

        print(f"[INFO] vision model: {vision_network}\n")

        config = ConfigCCIP(exp_name=args.exp_name, device=args.gpu, load_ckpt=args.ckpt_path)

        # CCIP model 

        clip = CLIP(image_encoder=image_encoder, cad_encoder=cad_encoder, dim_latent=config.dim_latent, multi_view=False, update_cad_grad=True).to(config.device)    
        count_params(clip)
                                
        # data loader: each batch contains: command, args, images and id

        train_loader = get_ccip_dataloader(phase="train", config=config, img_type='cad_image')
        val_loader = get_ccip_dataloader(phase="validation", config=config, img_type='cad_image')

        # optimizer

        optimizer = optim.AdamW(clip.parameters(), lr=config.lr)

        if config.use_scheduler: 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.1)
        else: 
            scheduler = None


        # trainer 

        ccip_trainer = TrainerCCIPModel(model=clip, config=config, optimizer=optimizer, scheduler=scheduler) 

        ccip_trainer.train(train_loader=train_loader, val_loader=val_loader, ckpt=args.ckpt_path)

        print_training_complete(save_path=config.exp_dir)

    elif model == "dp":

        with h5py.File(args.cad_embed_path, 'r') as f:
             cad_latent_data = f["zs"][:]

        # with h5py.File('data/image_embeddings.h5', 'r') as f:
        #     image_latent_data = f["zs"][:]

        with h5py.File(args.image_embed_path, 'r') as f:
            sketch_latent_data = f["zs"][:]

        cad_tensor = torch.tensor(cad_latent_data)
        sketch_tensor = torch.tensor(sketch_latent_data)

        dataset = DPDataset(cad_tensor, sketch_tensor)

        config = ConfigDP(exp_name=args.exp_name, device=0)

        model = ResNetDiffusion(d_in=config.d_in, n_blocks=config.n_blocks, d_main=config.d_main, 
                                d_hidden=config.d_hidden, dropout_first=config.dropout_first, 
                                dropout_second=config.dropout_second, d_out=config.d_out)

        diffusion = GaussianDiffusion1D(
        model,
        z_dim=config.z_dim,
        timesteps = config.timesteps,
        objective = config.objective, 
        auto_normalize=config.auto_normalize
        )


        trainer = Trainer1D(
        diffusion,
        dataset,
        config=config,
        device=config.device,
        train_batch_size = config.batch_size,
        train_lr = config.lr,
        train_num_steps = config.train_num_steps,         # total training steps
        gradient_accumulate_every = config.gradient_accumulate_every,    # gradient accumulation steps
        ema_decay = config.ema_decay,                # exponential moving average decay
        gt_data_path=args.cad_embed_path,
        amp = config.amp, 
        save_and_sample_every=config.save_and_sample_every
        )

        trainer.train()

    else:
        raise Exception("please choose between: autoencoder/ccip")
    


if __name__=="__main__": 
    parser = argparse.ArgumentParser(description="Train different models.")
    parser.add_argument("model", choices=["csr", "ccip", "dp"], help="Model to train")
    
    if "csr" in sys.argv:
        parser.add_argument("-name", "--exp_name", type=str, required=True, help="experiment numbers, keep fixed for the same experiment")
        parser.add_argument("-gpu", "--gpu", type=int, default=0, help="gpu device number, multi-gpu not supported")
        parser.add_argument("-ckpt", "--ckpt_path", type=str, default=None, help="path to checkpoint file", required=False)

    elif "ccip" in sys.argv: 
        parser.add_argument("-name", "--exp_name", type=str, required=True, help="experiment numbers, keep fixed for the same experiment")
        parser.add_argument("-gpu", "--gpu", type=int, default=0, help="gpu device number, multi-gpu not supported")
        parser.add_argument("-ckpt", "--ckpt_path", type=str, default=None, help="path to checkpoint file", required=False)
        parser.add_argument("-cad_ckpt", "--cad_ckpt_path", type=str, default=None, help="path to checkpoint file", required=True)

    elif "dp" in sys.argv: 
        parser.add_argument("-name", "--exp_name", type=str, required=True, help="experiment numbers, keep fixed for the same experiment")
        parser.add_argument("-gpu", "--gpu", type=int, default=0, help="gpu device number, multi-gpu not supported")
        parser.add_argument("-cad_emb", "--cad_embed_path", type=str, default=None, help="path to checkpoint file", required=True)
        parser.add_argument("-img_emb", "--image_embed_path", type=str, default=None, help="path to checkpoint file", required=True)
        
    args = parser.parse_args()
    train_model(args.model, args)
