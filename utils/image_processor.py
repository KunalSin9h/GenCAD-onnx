import numpy as np
from torch.utils.data import Dataset, DataLoader
from cadlib.macro import EOS_VEC
from multiprocessing import cpu_count
from PIL import Image
from torchvision import transforms
import cv2



class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


def process_image(image_path, img_type='image'): 

    apply_canny = CannyDetector()

    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(256),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                             std=[0.5, 0.5, 0.5]),
                    ])



    image = Image.open(image_path)

    if img_type == "image": 
        rgb_img = image.convert("RGB")
        img = np.array(rgb_img)
        l, h = 100, 200
        img = apply_canny(img, l, h)
        img = img[:, :, None]
        img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img)
        image_tensor = preprocess(img)

    elif img_type == "sketch": 
        rgb_img = image.convert("RGB")
        image_tensor = preprocess(rgb_img)

    return image_tensor.unsqueeze(0)
