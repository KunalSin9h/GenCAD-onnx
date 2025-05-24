# ----------------------------
# 
# Code borrowed from: https://github.com/ChrisWu1997/DeepCAD
#
#-----------------------------

from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
from cadlib.macro import EOS_VEC
import numpy as np
import random
from multiprocessing import cpu_count

from PIL import Image
from torchvision import transforms
import cv2
import einops



def cycle(dl):
    while True:
        for data in dl:
            yield data


def make_a_grid(occupancy_arr, color_numpy, voxel_resolution=32):
    voxel_resolution = int(voxel_resolution)
    cube_color = np.zeros(
        [voxel_resolution + 1, voxel_resolution + 1, voxel_resolution + 1, 4]
    )
    cube_color[occupancy_arr[:, 0], occupancy_arr[:, 1], occupancy_arr[:, 2], 0:3] = (
        color_numpy
    )
    cube_color[occupancy_arr[:, 0], occupancy_arr[:, 1], occupancy_arr[:, 2], 3] = 1
    return cube_color[:-1, :-1, :-1]


def get_voxel_data_json(voxel_file, voxel_resolution, device=torch.device('cpu')):
    file_base_name = os.path.basename(voxel_file).split(".")[0]

    with open(voxel_file, "r") as f:
        voxel_json = json.load(f)

    voxel_resolution = voxel_json["resolution"]
    occupancy_arr = np.array(voxel_json["occupancy"])
    color_numpy = np.array(voxel_json["color"])

    voxel_grid = make_a_grid(
        occupancy_arr,
        color_numpy,
        voxel_resolution=voxel_resolution,
    )[:, :, :, 3:4]

    voxel_grid = voxel_grid.transpose(3, 0, 1, 2).astype(np.float32)
    voxel_grid = torch.from_numpy(voxel_grid)
    # voxel_tensor = voxel_grid.unsqueeze(0)
    return voxel_grid



def smooth_edges(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)



def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = CADDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=is_shuffle, num_workers=cpu_count(),
                            pin_memory = True)
    return dataloader


def get_ccip_dataloader(phase, config, img_type="cad_image", multi_view=False, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    if multi_view:
        dataset = CCIPMultiViewDataset(phase, config, num_img=5)
    else: 
        dataset = CCIPDataset(phase, config, img_type=img_type)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=is_shuffle, num_workers=cpu_count(),
                            pin_memory = True)
    return dataloader


def get_ccip_dataloader_deterministic(phase, config, img_type="cad_image", shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = CCIPDatasetDeterministic(phase, config, img_type=img_type)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=is_shuffle, num_workers=cpu_count(),
                            pin_memory = True)
    return dataloader



def get_ldm_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = CADDataset(phase=phase, config=config)

    data_loader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=is_shuffle, num_workers=cpu_count(),
                            pin_memory = True)

    data_loader = cycle(data_loader)

    return data_loader


def get_ldm_cond_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = CADImageDataset(phase, config)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=is_shuffle, num_workers=cpu_count(),
                            pin_memory = True)
    
    # data_loader = cycle(data_loader)
    
    return data_loader


def get_diffusion_prior_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = CCIPDataset(phase, config)

    data_loader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=is_shuffle, num_workers=cpu_count(),
                            pin_memory = True)

    # data_loader = cycle(data_loader)

    return data_loader



def get_ccip_voxel_dataloader(phase, config, shuffle=None, augment=True):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = CCIPVoxelDataset(phase, config, augment=augment)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=is_shuffle, num_workers=cpu_count(),
                            pin_memory = True)
    return dataloader



class CADDataset(Dataset):
    def __init__(self, phase, config):
        super().__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.path = os.path.join(config.data_root, "filtered_data.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        self.max_n_loops = config.max_n_loops          # Number of paths (N_P)
        self.max_n_curves = config.max_n_curves            # Number of commands (N_C)
        self.max_total_len = config.max_total_len
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)


        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"command": command, "args": args, "id": data_id}

    def __len__(self):
        return len(self.all_data)



class CCIPDataset(Dataset):
    def __init__(self, phase, config, img_type="cad_image"):
        super().__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.img_type = img_type
        self.path = os.path.join(config.data_root, "filtered_data.json")
        self.img_id_path = os.path.join(config.data_root, "image_ids.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        with open(self.img_id_path, "r") as fp:
            self.img_ids = json.load(fp)

        if self.img_type == "cad_image":
            self.raw_image = os.path.join(config.data_root, "images")
        elif self.img_type == "cad_sketch":
            self.raw_image = os.path.join(config.data_root, "sketches")

        self.preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(256),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                             std=[0.5, 0.5, 0.5]),
                    ])


        # self.preprocess = transforms.Compose([
        #                         transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        #                         transforms.RandomRotation(degrees=(0, 180)),
        #                         transforms.Resize(256),
        #                         transforms.CenterCrop(256),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize(mean=[0.5, 0.5, 0.5], 
        #                                                 std=[0.5, 0.5, 0.5]),
        #                     ])
        
        # only needed for sketch image processing
        self.low_threshold, self.high_threshold = 100, 200
        self.apply_canny = CannyDetector()

        self.max_total_len = 60
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        img_id_by_data_id = self.img_ids[data_id]   # [0, 1, 2, 3, 4]

        # randomly choose one of the image id
        img_id = random.choice(img_id_by_data_id)
        image_path = os.path.join(self.raw_image, data_id + "_" + str(img_id) + ".png")
                
        image = Image.open(image_path)
        # gray_img = np.array(image.convert("L"))

        # # make edges more clear
        # edges = cv2.Canny(gray_img, 50, 150) 
        # kernel = np.ones((3, 3), np.uint8)
        # thick_edges = cv2.dilate(edges, kernel, iterations=1)

        if self.img_type == "cad_sketch":
            rgb_img = image.convert("RGB")  # remove transparent background
            img = np.array(rgb_img)
            l, h = 100, 200
            img = self.apply_canny(img, l, h)
            # img = smooth_edges(img)
            img = img[:, :, None]
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img)
            image_tensor = self.preprocess(img)

        else:
            # enhanced_img = cv2.addWeighted(gray_img, 1.0, thick_edges, 0.5, 0)
            # enhanced_img_3channel = np.stack([enhanced_img]*3, axis=-1)
            # enhanced_img_pil = Image.fromarray(enhanced_img_3channel)
            # image_tensor = self.preprocess(enhanced_img_pil)
            rgb_img = image.convert("RGB")  # remove transparent background
            image_tensor = self.preprocess(rgb_img)

        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"command": command, "args": args, "image":image_tensor, "id": data_id}

    def __len__(self):
        return len(self.all_data)
    


class CCIPDatasetDeterministic(Dataset):
    def __init__(self, phase, config, img_type="cad_image"):
        super().__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.img_type = img_type
        self.path = os.path.join(config.data_root, "filtered_data.json")
        self.img_id_path = os.path.join(config.data_root, "image_ids.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        with open(self.img_id_path, "r") as fp:
            self.img_ids = json.load(fp)

        if self.img_type == "cad_image":
            self.raw_image = os.path.join(config.data_root, "images")
        elif self.img_type == "cad_sketch":
            self.raw_image = os.path.join(config.data_root, "sketches")

        self.preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(256),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                             std=[0.5, 0.5, 0.5]),
                    ])

        # only needed for sketch image processing
        self.low_threshold, self.high_threshold = 100, 200
        self.apply_canny = CannyDetector()


        self.max_total_len = 60
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        img_id_by_data_id = self.img_ids[data_id]   # [0, 1, 2, 3, 4]

        # only choose the first of the image id
        img_id = img_id_by_data_id[0]
        image_path = os.path.join(self.raw_image, data_id + "_" + str(img_id) + ".png")
                
        image = Image.open(image_path)
        # gray_img = np.array(image.convert("L"))

        # # make edges more clear
        # edges = cv2.Canny(gray_img, 50, 150) 
        # kernel = np.ones((3, 3), np.uint8)
        # thick_edges = cv2.dilate(edges, kernel, iterations=1)

        if self.img_type == "cad_sketch":
            rgb_img = image.convert("RGB")  # remove transparent background
            img = np.array(rgb_img)
            l, h = 100, 200
            img = self.apply_canny(img, l, h)
            img = img[:, :, None]
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img)
            image_tensor = self.preprocess(img)

        else:
            # enhanced_img = cv2.addWeighted(gray_img, 1.0, thick_edges, 0.5, 0)
            # enhanced_img_3channel = np.stack([enhanced_img]*3, axis=-1)
            # enhanced_img_pil = Image.fromarray(enhanced_img_3channel)
            # image_tensor = self.preprocess(enhanced_img_pil)
            rgb_img = image.convert("RGB")  # remove transparent background
            image_tensor = self.preprocess(rgb_img)

        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"command": command, "args": args, "image":image_tensor, "id": data_id}

    def __len__(self):
        return len(self.all_data)
    


class CCIPMultiViewDataset(Dataset):
    def __init__(self, phase, config, num_img=5):
        super().__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.num_img = num_img
        self.path = os.path.join(config.data_root, "multiview_data.json")
    
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        self.raw_image = os.path.join(config.data_root, "multi_view_images")

        self.preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(256),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                             std=[0.5, 0.5, 0.5]),
                    ])

        # -- alternative image augmentation

        # self.preprocess = transforms.Compose([
        #                         transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        #                         transforms.RandomRotation(degrees=(0, 180)),
        #                         transforms.Resize(256),
        #                         transforms.CenterCrop(256),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize(mean=[0.5, 0.5, 0.5], 
        #                                                 std=[0.5, 0.5, 0.5]),
        #                     ])
        
        self.max_total_len = 60
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")

        # randomly choose one of the image id
        random_indices = random.sample(range(21), self.num_img)
        image_paths = [os.path.join(self.raw_image, f"{data_id}_{str(idx).zfill(2)}.png") for idx in random_indices]

        images = [] 
        for img_path in image_paths:  
            image = Image.open(img_path)
            rgb_img = image.convert("RGB")
            img_tensor = self.preprocess(rgb_img)
            images.append(img_tensor)

        image_tensor = torch.cat(images, dim=0)

        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"command": command, "args": args, "image":image_tensor, "id": data_id}

    def __len__(self):
        return len(self.all_data)



class CCIPVoxelDataset(Dataset):
    def __init__(self, phase, config, augment=True):
        super().__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.path = os.path.join(config.data_root, "voxel_data.json")
        self.augment = augment
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]
        

        self.voxel_id_path = os.path.join(config.data_root, "voxel_data.json")
        with open(self.voxel_id_path, "r") as fp:
            self.voxel_ids = json.load(fp)[phase]

        self.raw_voxel = os.path.join(config.data_root, "voxels")

        self.max_total_len = 60
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.voxel_ids[index]
        subdir_name, id_name = data_id.split('/')[0], data_id.split('/')[-1]

        # cad data
        h5_path = os.path.join(self.raw_data,  data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)

        # voxel data
        voxel_file = os.path.join(self.raw_voxel, subdir_name + '/32/' + id_name + ".json")
        voxel_tensor = get_voxel_data_json(voxel_file=voxel_file, voxel_resolution=32)

        if self.augment and self.phase == "train":
            voxel_tensor = self.augment_voxel(voxel_tensor)

        return {"command": command, "args": args, "voxel":voxel_tensor, "id": data_id}

    def __len__(self):
        return len(self.all_data)

    def augment_voxel(self, voxel):
        """
        Apply random 90-degree rotation to the voxel tensor.
        :param voxel: Tensor of shape (1, 32, 32, 32)
        :return: Rotated voxel tensor
        """
        k = random.choice([0, 1, 2, 3])  # Number of 90-degree rotations
        axis = random.choice([(2, 3), (1, 3), (1, 2)])  # Random rotation axis
        voxel = voxel.rot90(k=k, dims=axis)

        if random.random() < 0.5:  # 50% chance of shifting
            shift_voxel = torch.zeros_like(voxel)  # Create empty voxel grid
            shift_x = random.randint(-3, 3)  # Random shift between -3 to +3
            shift_y = random.randint(-3, 3)
            shift_z = random.randint(-3, 3)

            # Compute slicing indices
            x_start, x_end = max(0, shift_x), min(32, 32 + shift_x)
            y_start, y_end = max(0, shift_y), min(32, 32 + shift_y)
            z_start, z_end = max(0, shift_z), min(32, 32 + shift_z)

            shift_voxel[..., x_start:x_end, y_start:y_end, z_start:z_end] = \
                voxel[..., :x_end - x_start, :y_end - y_start, :z_end - z_start]
            
            voxel = shift_voxel  # Update with shifted voxel

        return voxel



class LDMDataset(Dataset):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()


class CADImageDataset(Dataset):
    def __init__(self, phase, config, img_type="cad_image"):
        super().__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.img_type = img_type
        self.path = os.path.join(config.data_root, "filtered_data.json")
        self.img_id_path = os.path.join(config.data_root, "image_ids.json")

        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        with open(self.img_id_path, "r") as fp:
            self.img_ids = json.load(fp)

        if self.img_type == "cad_image":
            self.raw_image = os.path.join(config.data_root, "images")
        elif self.img_type == "cad_sketch":
            self.raw_image = os.path.join(config.data_root, "sketches")

        self.preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(256),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                             std=[0.5, 0.5, 0.5]),
                    ])

        self.max_n_loops = config.max_n_loops          # Number of paths (N_P)
        self.max_n_curves = config.max_n_curves            # Number of commands (N_C)
        self.max_total_len = config.max_total_len
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        img_id_by_data_id = self.img_ids[data_id]   # [0, 1, 2, 3, 4]

        # randomly choose one of the image id
        img_id = random.choice(img_id_by_data_id)
        image_path = os.path.join(self.raw_image, data_id + "_" + str(img_id) + ".png")

        image = Image.open(image_path)
        rgb_img = image.convert("RGB")  # remove transparent background
        image_tensor = self.preprocess(rgb_img)

        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"command": command, "args": args, "image":image_tensor, "id": data_id}

    def __len__(self):
        return len(self.all_data)
    

# ---- LDM Dataset ----

class DPDataset(Dataset):
    def __init__(self, cad_tensor, image_tensor):
        super().__init__()
        self.cad_tensor = cad_tensor.clone()
        self.image_tensor = image_tensor.clone()
        
    def __len__(self):
        return len(self.cad_tensor)

    def __getitem__(self, idx):
        return self.cad_tensor[idx].clone(),  self.image_tensor[idx].clone()

    



