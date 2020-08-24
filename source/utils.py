"""
Module with utility functions like changing annotation format
"""

import os
import sys
import random
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def xyxy_to_xywh(df):
    # Convert from [x_min, y_min, x_max, y_max] to [x, y, w, h] 
    df['x'] = df['x_min']
    df['y'] = df['y_min']
    df['w'] = df['x_max'] - df['x_min']
    df['h'] = df['y_max'] - df['y_min']
    return df.drop(['x_min', 'y_min', 'x_max', 'y_max'], axis=1)


def plot_image(index, dataset_retriever, size=(8, 8)):
    image, target, image_id = dataset_retriever[index]
    boxes = target['boxes'].cpu().numpy().astype(np.int32)
    numpy_image = image.permute(1,2,0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=size)
    for box in boxes:
        cv2.rectangle(numpy_image,
            (box[1], box[0]), (box[3],  box[2]), (1, 0, 0), 1)

    ax.set_axis_off()
    ax.imshow(numpy_image)


def set_environment_configs(seed=20):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    sys.path.insert(0, "source/pytorch-efficientdet")
    sys.path.insert(0, "source/wbf")
