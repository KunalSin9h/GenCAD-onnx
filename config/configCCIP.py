import os
from datetime import datetime
import shutil
import sys 

class ConfigCCIP:
    """
    Configuration of the autoencoder
    """
    def __init__(self, exp_name="test_model", 
                    phase="train", num_epochs=300, 
                    lr=1e-3, batch_size=64,
                    load_ckpt=False, 
                    save_every=25, 
                    val_every=5, 
                    device=None, 
                    image_model='resnet_18', 
                    dim_latent=256, 
                    overwrite=False):
        
        self.is_train = phase == "train"

        # --------- experiment parameters ------------
        self.exp_name = exp_name + "/CCIP"
        self.load_ckpt = load_ckpt
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.val_every = val_every
        self.batch_size = batch_size  
        self.lr = lr  
        self.use_scheduler = True
        self.num_workers = 12   
        self.device = device
        self.image_model = image_model             
        self.overwrite = overwrite
        self.dim_latent = dim_latent

        # experiment paths  
        # experiment directory: results/experiment_name  
        self.proj_dir = "results"  
        self.data_root = "data/"

        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)  
        self.model_dir = os.path.join(self.exp_dir, 'trained_models')        # results/experiment_name/log
        self.log_dir = os.path.join(self.exp_dir, 'log')            # results/experiment_name/log

        # check if directories exit
        if self.overwrite:
            if os.path.exists(self.exp_dir):
                response = input(f"Directory '{self.exp_dir}' already exists. Overwrite? (y/n): ").strip().lower()
                if response == 'y':
                    print("Overwriting the existing directory.")
                    shutil.rmtree(self.exp_dir)
                else: 
                    sys.exit("exit") 

        # create directories
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # ----------- model hyperparameters -----------

        self.write_config()

    def write_config(self):
        # save this configuration
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.is_train:
            config_path = os.path.join(self.exp_dir, f'config_clip_model.txt')
            with open(config_path, 'w') as f:
                    f.write(f'Config for # # clip model # # ' + '\n')
                    f.write(f'Time -> {current_time}' + '\n' + '\n')
                    f.write(f'  experiment directory -> {self.proj_dir}/{self.exp_name}/' + '\n')
                    f.write(f'  data directory -> {self.data_root}' + '\n')
                    f.write('# '*25 + '\n' + '\n')
                    f.write(f'  number of epochs: {self.num_epochs}' + '\n')
                    f.write(f'  batch size: {self.batch_size}' + '\n')
                    f.write(f'  learning rate: {self.lr}' + '\n')
                    f.write(f'  device: {self.device}' + '\n' + '\n')

                    f.write(f'  image model: {self.image_model}' + '\n' + '\n')
                    f.write(f'  save every: {self.save_every}' + '\n')
                    f.write(f'  validate every: {self.val_every}' + '\n' + '\n')
                    
                    f.write('# '*25 + '\n' + '\n')
