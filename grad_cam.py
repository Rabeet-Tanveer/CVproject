import argparse
import os
import numpy as np
import torch
from torchvision import models
from PIL import Image
from lib.gradients import (
    VanillaGrad,
    SmoothGrad,
    GuidedBackpropGrad,
    GuidedBackpropSmoothGrad
)
from lib.image_utils import preprocess_image, save_as_gray_image
from lib.labels import IMAGENET_LABELS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--img', type=str, default='', help='Input image path')
    parser.add_argument('--out_dir', type=str, default='/content/pytorch-smoothgrad/results/grad/', help='Result directory path')
    parser.add_argument('--n_samples', type=int, default=10, help='Sample size for SmoothGrad')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    
    device = "GPU" if args.cuda else "CPU"
    print(f"Using {device} for computation")
    print(f"Output directory: {args.out_dir}")
    print(f"Sample size for SmoothGrad: {args.n_samples}\n")
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

    # Load and prepare image
    if args.img:
        img_np = load_image(args.img)
    else:
        print("No image provided. Loading fallback image (e.g., astronaut from skimage)...")
        from skimage.data import astronaut
        img_np = astronaut()
        img_np = cv2.resize(img_np, (224, 224))

    preprocessed_img = preprocess_image(img_np, cuda=args.cuda)

    # Load model
    model = models.vgg19(pretrained=True)
    if args.cuda:
        model = model.cuda()

    # Prediction
    output = model(preprocessed_img)
    pred_index = torch.argmax(output).item()
    print(f"Prediction: {IMAGENET_LABELS[pred_index]}")

    # Compute and save Vanilla Grad
    vanilla_grad = VanillaGrad(model, cuda=args.cuda)
    vanilla_saliency = vanilla_grad(preprocessed_img, index=pred_index)
    save_as_gray_image(vanilla_saliency, os.path.join(args.out_dir, 'vanilla_grad.jpg'))
    print("Saved Vanilla Grad")

    # Guided Backpropagation
    guided_grad = GuidedBackpropGrad(model, cuda=args.cuda)
    guided_saliency = guided_grad(preprocess_image(img_np, args.cuda), index=pred_index)
    save_as_gray_image(guided_saliency, os.path.join(args.out_dir, 'guided_grad.jpg'))
    print("Saved Guided Backprop")

    # SmoothGrad
    smooth_grad = SmoothGrad(model, cuda=args.cuda, n_samples=args.n_samples, magnitude=True)
    smooth_saliency = smooth_grad(preprocess_image(img_np, args.cuda), index=pred_index)
    save_as_gray_image(smooth_saliency, os.path.join(args.out_dir, 'smooth_grad.jpg'))
    print("Saved SmoothGrad")

    # Guided SmoothGrad
    guided_smooth_grad = GuidedBackpropSmoothGrad(model, cuda=args.cuda, n_samples=args.n_samples, magnitude=True)
    guided_smooth_saliency = guided_smooth_grad(preprocess_image(img_np, args.cuda), index=pred_index)
    save_as_gray_image(guided_smooth_saliency, os.path.join(args.out_dir, 'guided_smooth_grad.jpg'))
    print("Saved Guided SmoothGrad")

if __name__ == '__main__':
    main()
