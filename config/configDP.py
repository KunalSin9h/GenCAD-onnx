import os
from datetime import datetime
import shutil
import sys 

from cadlib.macro import ARGS_DIM, N_ARGS, ALL_COMMANDS, MAX_N_EXT, \
                        MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN


class ConfigDP:
    """
    Configuration of the autoencoder
    """
    def __init__(self, exp_name="test_model", 
                    phase="train", 
                    load_ckpt=False, 
                    device=None, 
                    overwrite=False):
        
        self.is_train = phase == "train"

        # --------- experiment parameters ------------
        self.exp_name = exp_name + "/DP"
        self.load_ckpt = load_ckpt
        self.batch_size = 2048
        self.lr = 1e-5
        self.train_num_steps = 10
        self.gradient_accumulate_every = 2
        self.ema_decay = 0.995
        self.amp = True
        self.save_and_sample_every = 5
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
        self.d_in = 256
        self.d_out = 256
        self.z_dim = 256
        self.n_blocks = 10 
        self.d_main = 2048
        self.d_hidden = 2048
        self.dropout_first = 0.1
        self.dropout_second = 0.1
        self.d_out = 256

        self.timesteps = 500
        self.objective = 'pred_x0'
        self.auto_normalize = False


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
                    f.write(f'  device: {self.device}' + '\n' + '\n')

        