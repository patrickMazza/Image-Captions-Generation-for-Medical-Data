import json
import cv2
import glob
import numpy as np
from skimage import io
import pandas as pd
import os

import torch
from torchvision import transforms


def caption_to_figure(caption_dict):
    figure_file_name = '_'.join([caption_dict['pdf_hash'], caption_dict['fig_uri']])
    return figure_file_name


def array_to_tensor(img_array) -> torch.FloatTensor:
    return torch.FloatTensor(img_array / 255)


def preprocess_image(img, target_size=256):
    # normalize, pad, resize
    img_max_size = max(img.shape)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Pad(img_max_size, fill=0),
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size)
    ])
    img = transform(img)

    return img


if __name__ == '__main__':
    figures_data = pd.read_csv('/Users/mariadobko/Documents/Cornell/release/Cleaned_CT_cluster_data.csv')
    example = figures_data.iloc[0]
    im_path = os.path.join('/Users/mariadobko/Documents/Cornell/release/figures/', caption_to_figure(example))
    image = io.imread(im_path)

    result = preprocess_image(image)
    print(image.shape, result.shape)
