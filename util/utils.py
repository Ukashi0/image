from PIL import Image
import inspect
import re
import numpy as np
import torch
import os
import collections
from torch.optim import lr_scheduler
import torch.nn.init as init


def tensor2im(img_tensor, type=np.uint8):
    img_np = img_tensor[0].cpu().float().numpy()
    img_np = (np.transpose(img_np, (1, 2, 0)) + 1) / 2.0 * 255.0
    img_np = np.maximum(img_np, 0)
    img_np = np.minimum(img_np, 255)
    return img_np.astype(type)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
