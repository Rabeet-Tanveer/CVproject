import argparse
import os
import numpy as np
import cv2
import torch
from torchvision import models
from PIL import Image
from lib.gradients import GradCam, GuidedBackpropGrad
from lib.image_utils import preprocess_image, save_cam_image, save_as_gray_image
from lib.labels import IMAGENET_LABELS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--img', type=str, default='', help='Input image path')
    parser.add_argument('--out_dir', type=str, default='/content/pytorch-smoothgrad/results/cam/', help='Result directory path')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    
    print(f"Using {'GPU' if args.cuda else 'CPU'} for computation")
    print(f"Output directory: {args.out_dir}\n")
    
    return args

def load_image(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {img_path}: {e}")
    return np.array(img.resize((224, 224)))

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    target_layer_names = ['35']
    target_index = None

    # Load and prepare input image
    if args.img:
        img_np = load_image(args.img)
    else:
        from skimage.data import astronaut
        img_np = cv2.resize(astronaut(), (224, 224))

    preprocessed_img = preprocess_image(img_np, cuda=args.cuda)

    # Load model
    model = models.vgg19(pretrained=True)
    if args.cuda:
        model = model.cuda()

    # Prediction
    output = model(preprocessed_img)
    pred_index = torch.argmax(output).item()
    print(f'Prediction: {IMAGENET_LABELS[pred_index]}')

    # Grad-CAM
    grad_cam = GradCam(pretrained_model=model, target_layer_names=target_layer_names, cuda=args.cuda)
    mask = grad_cam(preprocessed_img, target_index)

    save_cam_image(img_np / 255.0, mask, os.path.join(args.out_dir, 'grad_cam.jpg'))
    print('Saved Grad-CAM image')

    # Guided Backpropagation
    preprocessed_img = preprocess_image(img_np, cuda=args.cuda)
    guided_backprop = GuidedBackpropGrad(pretrained_model=model, cuda=args.cuda)
    guided_backprop_saliency = guided_backprop(preprocessed_img, index=target_index)

    # Combine Grad-CAM and Guided Backpropagation
    cam_mask = np.repeat(mask[np.newaxis, :, :], 3, axis=0)  # Shape (3, H, W)
    cam_guided_backprop = cam_mask * guided_backprop_saliency
    save_as_gray_image(cam_guided_backprop, os.path.join(args.out_dir, 'guided_grad_cam.jpg'))
    print('Saved Guided Grad-CAM image')

if __name__ == '__main__':
    main()
