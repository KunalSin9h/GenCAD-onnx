import os
from datetime import datetime
import shutil
import sys 

from cadlib.macro import ARGS_DIM, N_ARGS, ALL_COMMANDS, MAX_N_EXT, \
                        MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN


class ConfigAE:
    """
    Configuration of the autoencoder
    """
    def __init__(self, exp_name="test_model", 
                    phase="train", num_epochs=10, 
                    lr=1e-3, batch_size=2,
                    load_ckpt=False, 
                    save_every=50, 
                    val_every=10, 
                    device=None, 
                    overwrite=False):
        
        self.is_train = phase == "train"

        # --------- experiment parameters ------------
        self.exp_name = exp_name + "/CSR"
        self.load_ckpt = load_ckpt
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.val_every = val_every
        self.batch_size = batch_size
        self.lr = lr
        self.use_scheduler = True
        self.warmup_step = 2000   
        self.num_workers = 12   
        self.device = device
        self.overwrite = overwrite
        
             
        # experiment paths
        # experiment directory: results/experiment_name
        self.proj_dir = "results"
        self.data_root = "data"

        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        self.model_dir = os.path.join(self.exp_dir, 'trained_models')        # results/experiment_name/log
        self.log_dir = os.path.join(self.exp_dir, 'log')            # results/experiment_name/log


        if self.overwrite:
            if os.path.exists(self.exp_dir):
                response = input(f"Directory '{self.exp_dir}' already exists. Overwrite? (y/n): ").strip().lower()
                if response == 'y':
                    print("Overwriting the existing directory.")
                    shutil.rmtree(self.exp_dir)
                else: 
                    sys.exit("exit")  

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


        # ----------- model hyperparameters -----------
        self.n_enc_heads = 8                 # Transformer config: number of heads
        self.n_enc_layers = 4                # Number of Encoder blocks
        self.n_dec_heads = 8                 # Transformer config: number of heads
        self.n_dec_layers = 4                # Number of Encoder blocks        
        self.dim_feedforward = 512       # Transformer config: FF dimensionality
        self.d_model = 256               # Transformer config: model dimensionality
        self.dropout = 0.1                # Dropout rate used in basic layers and Transformers
        self.dim_z = 256                 # Latent vector dimensionality
        self.use_group_emb = True

        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0
        }
        self.max_num_groups = 30
        self.grad_clip = 1.0


        # ---------- CAD parameters: fixed ------------
        # ----- Config file is not updated with these information because they are fixed for all experiments

        self.args_dim = ARGS_DIM # 256
        self.n_args = N_ARGS
        self.n_commands = len(ALL_COMMANDS)  # line, arc, circle, EOS, SOS

        self.max_n_ext = MAX_N_EXT
        self.max_n_loops = MAX_N_LOOPS
        self.max_n_curves = MAX_N_CURVES

        self.max_num_groups = 30
        self.max_total_len = MAX_TOTAL_LEN

        if self.overwrite:
            self.write_config()

    def write_config(self):
        # save this configuration
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.is_train:
            config_path = os.path.join(self.exp_dir, f'config_autoencoder.txt')
            with open(config_path, 'w') as f:
                    f.write(f'Config for # # autoencoder model # # ' + '\n')
                    f.write(f'Time -> {current_time}' + '\n' + '\n')
                    f.write(f'  experiment directory -> {self.proj_dir}/{self.exp_name}/' + '\n')
                    f.write(f'  data directory -> {self.data_root}/' + '\n')
                    f.write(f'  load chekpoint: {self.load_ckpt}' + '\n' + '\n')
                    f.write(f'  num of workers: {self.num_workers}' + '\n')
                    f.write(f'  device: {self.device}' + '\n' + '\n')

                    f.write('\n' + '# '*25 + '\n')
                    f.write(f'  number of epochs: {self.num_epochs}' + '\n')
                    f.write(f'  batch size: {self.batch_size}' + '\n')
                    f.write(f'  learning rate: {self.lr}' + '\n')
                    f.write(f'  scheduler use: {self.use_scheduler}' + '\n')
                    f.write(f'        warmup steps: {self.warmup_step}' + '\n')
                    f.write(f'  dropout: {self.dropout}' + '\n')
                    f.write(f'  loss weights:' + '\n')
                    f.write(f'        command loss: {self.loss_weights["loss_cmd_weight"]}' + '\n')
                    f.write(f'        argument loss: {self.loss_weights["loss_args_weight"]}' + '\n')
                    
                    f.write(f'  save every: {self.save_every}' + '\n')
                    f.write(f'  validate every: {self.val_every}' + '\n' + '\n')
                    
                    f.write('\n' + '# '*25 + '\n')
                    f.write(f'  encoder -> heads: {self.n_enc_heads}, layers: {self.n_enc_layers}' + '\n')
                    f.write(f'  decoder -> heads: {self.n_dec_heads}, layers: {self.n_dec_layers}' + '\n')
                    f.write(f'  latent dimension: {self.dim_z}' + '\n')

                    f.write(f'  use group embedding: {self.use_group_emb}' + '\n')
                    f.write(f'        number of groups: {self.max_num_groups}' + '\n')

        