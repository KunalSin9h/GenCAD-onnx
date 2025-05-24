import numpy as np
import random
import time
import copy
import os
import shutil
import glob 
import json
from collections import OrderedDict
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms

from utils.model_utils import AvgMeter
from utils.cad_dataset import get_dataloader, get_ccip_dataloader


class TestCCIPModel: 
    def __init__(self, model, ckpt_path, config): 
        self.model = model
        self.batch_size = config.batch_size
        self.device = config.device
        self.exp_dir = config.exp_dir
        self.log_dir = config.log_dir
        self.model_path = config.model_dir

        self.ckpt_path = ckpt_path

        self.preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

        # load checkpoint
        self._load_ckpt()


    def _load_ckpt(self, image_encoder=True):
        if not os.path.exists(self.ckpt_path):
            raise ValueError("Checkpoint {} not exists.".format(self.ckpt_path))

        checkpoint = torch.load(self.ckpt_path)
        print('# '* 25)
        print("Loading Image encoder checkpoint from: {} ...".format(self.ckpt_path))


        if image_encoder:
            # load model 
            self.model.image_encoder.load_state_dict(checkpoint['model_state_dict'])

        print('\n # # # # # Image encoder checkpoint loaded')
        

    def cad_prob_from_img(self, commands, args, image, return_probs=False):
        """
        images: (B, 3, 448, 448)
        cad_databases: collection of h5 files
        """

        with torch.no_grad(): 
            self.cad_embeddings = self.model.encode_cad(commands, args)
            self.img_embeddings = self.model.encode_image(image)

            image_logits, cad_logits = self.model.get_logits(commands, args, image)
            probs = image_logits.softmax(dim=-1).cpu().numpy()

            cad_indices = np.argmax(probs, axis=1)

        if return_probs: 
            return cad_indices, probs

        return cad_indices


    def load_image(self, image_path):
        image = Image.open(image_path)
        rgb_img = image.convert("RGB")  # remove transparent background
        image_tensor = self.preprocess(rgb_img)

        return image_tensor.unsqueeze(0)

    def _extract_numeric(self, value):
        return int(value.split('/')[-1])

    def _format_output(self, image_paths, probs, k): 

        row_img_array = np.vectorize(self._extract_numeric)(np.array(image_paths))
        col_img_array = np.vectorize(self._extract_numeric)(np.array(image_paths))

        # Initialize the prob_array and fill it as required
        prob_array = np.array(probs)

        # Adding appropriate labels to rows and columns
        column_labels = [''] + [f'{int(x):08d}' for x in col_img_array]
        row_labels = [f'{int(x):08d}' for x in row_img_array]

        # Create a function to format each row
        def format_row(row, labels=None):
            if labels is not None:
                return " | ".join(f"{label:<10}" for label in labels)
            return " | ".join(f"{value:<10.8f}" for value in row)

        # Write the header
        output_path = f'{self.exp_dir}/retrieved_images/{k}/probs.txt'
        with open(output_path, 'w') as f:
            # Write the column headers
            f.write(format_row(None, column_labels) + "\n")
            
            # Write each row with its row label
            for label, row in zip(row_labels, prob_array):
                f.write(f"{label:<10} | " + format_row(row) + "\n")

    def calculate_retrieval_score(self, num_exp, data_loader, 
                                random_eval=False, save_output=False):
        """
        num_exp: number of evaluations 
        data_loader: test data loader
        random_eval: eval random guess 
        save_output: saves output as a nice formatted text file 
        """ 
        success = 0 

        pbar = tqdm(range(num_exp))

        retrieval_res = {}

        for k in pbar:

            batch = next(iter(data_loader))

            commands = batch["command"].to(self.device) # (B, 60)
            args = batch["args"].to(self.device)  # (B, 60, 16)
            ids = batch["id"]

            image_paths = ["data/images/" + str(img_id) for img_id in ids] 

            image_tensors = []
            img_path_ext, input_images = [], []
            for idx, img_path in enumerate(image_paths): 
                pattern = img_path + "_*.png"
                # Use glob to find all files matching the pattern
                matching_files = glob.glob(pattern)
                rand_img = random.choice(matching_files)
                img_tensor = self.load_image(rand_img)
                image_tensors.append(img_tensor)

                img_path_ext.append(sorted(matching_files)[0])
                input_images.append(rand_img)

            destination_folder_1 = f"{self.exp_dir}/retrieved_images/{k}/cad_images"
            destination_folder_2 = f"{self.exp_dir}/retrieved_images/{k}/input_images"

            if os.path.exists(destination_folder_1):
                shutil.rmtree(destination_folder_1)
            if os.path.exists(destination_folder_2):
                shutil.rmtree(destination_folder_2)

            os.makedirs(destination_folder_1)
            os.makedirs(destination_folder_2)

            # Copy each image to the new folder
            for img in img_path_ext:
                shutil.copy(img, destination_folder_1)

            for img in input_images:
                shutil.copy(img, destination_folder_2)

            image_tensors = torch.cat(image_tensors, dim=0).to(self.device)

            cad_indices, probs = self.cad_prob_from_img(commands, args, image_tensors, return_probs=True)
            rand_cad_indices = [random.randint(min(cad_indices), max(cad_indices)) for _ in cad_indices]

            if random_eval:
                eval_indices = rand_cad_indices
            else: 
                eval_indices = cad_indices

            pred_ids = [ids[k] for k in eval_indices]
            res = [1 if ids[k] == pred_ids[k] else 0 for k in range(len(ids))]
            retrievd = sum(res)
            success += retrievd
            pbar.set_description("Retrieved #{} --> {}/{}".format(k, retrievd, self.batch_size))
            
            retrieval_res[k] = "Retrieved #{} --> {}/{}".format(k, retrievd, self.batch_size)

            if save_output:
                self._format_output(image_paths, probs, k)


        retrieval_res[k+1] = f'Total CAD retrieval percentage: {success / (self.batch_size * num_exp)}'

        with open(f'{self.exp_dir}/retrieved_images/res.json', 'w') as json_file:
            json.dump(retrieval_res, json_file, indent=4)

        print(f'Total CAD retrieval percentage: {success / (self.batch_size * num_exp)}')