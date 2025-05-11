import cv2
import numpy as np
import torch
from torch.autograd import Variable

def preprocess_image(img, cuda=False):
    """
    Convert input image (HWC, RGB [0–255]) to normalized PyTorch tensor (1x3xHxW)
    """
    # Convert BGR to RGB and normalize
    means = [0.485, 0.456, 0.406]
    stds  = [0.229, 0.224, 0.225]

    img = img[:, :, ::-1] / 255.0  # BGR to RGB and scale
    img = (img - means) / stds
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add batch dim

    if cuda:
        img_tensor = img_tensor.cuda()

    img_tensor.requires_grad = True
    return img_tensor


def save_as_gray_image(img, filename, percentile=99):
    """
    Save a 3xHxW gradient map as a grayscale heatmap
    """
    img = np.abs(img)  # remove sign
    img_2d = np.sum(img, axis=0)  # convert to HxW

    vmax = np.percentile(img_2d, percentile)
    img_norm = np.clip(img_2d / (vmax + 1e-8), 0, 1)  # avoid divide-by-zero
    img_uint8 = (img_norm * 255).astype(np.uint8)

    cv2.imwrite(filename, img_uint8)


def save_cam_image(img, mask, filename):
    """
    Save CAM result overlaid on original image
    img: original image as float32 HxWxC (range 0–1)
    mask: heatmap 2D array (range 0–1)
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam + 1e-8)  # normalize

    cam_uint8 = np.uint8(255 * cam)
    cv2.imwrite(filename, cam_uint8)
