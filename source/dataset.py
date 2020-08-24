"""
Module for loading and augmenting images / bounding boxes.
The following augmentation techniques are being used:

- Cutout, Random Crop and Flip
- Change brightness, saturation, color and blur
- Cutmix technique
- Mixup technique
"""

import os
import random
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2


class DatasetRetriever(Dataset):
    """Class for loading and augmenting images and corresponding boxes.

    When using the parameter test=False, it will apply Cutmix and Mixup.
    The transforms passed as a parameter will always be applied.

    Parameters:
        df: a frame with x, y, w, h columns for each bbox
        image_ids: an array with each image id as a string
        images_path: path to the directory containing the images
        transforms: A transformer/composer function (albumentation library)
    """
    def __init__(self, df, image_ids, images_path, transforms=None, test=False):
        super().__init__()
        self.df = df
        self.image_ids = image_ids
        self.transforms = transforms
        self.test = test
        self.root_path = images_path

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        if self.test or random.random() > 0.6:
            image, boxes = self.load_image_and_boxes(index)
        else:
            if random.random() > 0.5:
                image, boxes = self.load_cutmix_image_and_boxes(index)
            else:
                image, boxes = self.load_mixup_image_and_boxes(index)

        target = {
            'boxes': boxes,
            'image_id': torch.tensor([index]),
            'labels': torch.ones((boxes.shape[0],), dtype=torch.int64) # single class
        }

        if self.transforms:
            for i in range(50):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': target['labels']
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor,
                        zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]
                    break
            else:
                raise RuntimeError(f'No bounding boxes for {image_id} after 50 tries')
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
    def random_indexes(self, num_indexes): # return a list of rnd image's indexes
        return [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(num_indexes)]

    def load_image_and_boxes(self, index):
        """Load one image from disk and the corresponding bounding boxes."""
        image_id = self.image_ids[index]

        image_path = os.path.join(self.root_path, image_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.df[self.df['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """ Cutmix implementation (Yun et al, 2019)."""
        w, h = imsize, imsize
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + self.random_indexes(num_indexes=3)

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # large image
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # small image
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh
            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes
    
    def load_mixup_image_and_boxes(self, index, mixup_coef=0.5, box_threshold=0.3):
        """Simple Mixup implementation (Zhang et al)
        
        Linear interpolation between two images for augmentation:
        new_x = λxi + (1 − λ)xj

        where xi and xj are n-dimensional vectors (images in this case)

        Arguments:
            index: original image index from the training data
            mixup_coef: how much to mix from random image
            box_threshold: if mixup_coef is above this level, removes
            all the bounding boxes from the other image
        """
        image, boxes = self.load_image_and_boxes(index)
        r_index = self.random_indexes(num_indexes=1)[0]
        r_image, r_boxes = self.load_image_and_boxes(r_index)

        mixup_image = mixup_coef * r_image + (1 - mixup_coef) * image
        if mixup_coef < box_threshold:
            mixup_boxes = boxes
        else:
            mixup_boxes = np.concatenate([boxes, r_boxes])
        return mixup_image, mixup_boxes


class ImageRetriever(Dataset):
    """Class that loads images for doing predictions"""
    def __init__(self, image_ids, images_path, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.transforms = transforms
        self.root_path = images_path

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.root_path, image_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


def get_train_transforms(height=1024, width=1024):
    """Data Algumentation and resize for the training dataset"""
    rnd_cutout = A.OneOf([
        A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.6),
        A.Cutout(num_holes=16, max_h_size=32, max_w_size=32, fill_value=0, p=0.3),
        A.Cutout(num_holes=64, max_h_size=16, max_w_size=16, fill_value=0, p=0.1),
    ], p=0.5)
    
    rnd_effect = A.OneOf([
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
            val_shift_limit=0.2, p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
    ], p=0.9)
    
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(740, 840), height=1024, width=1024, p=0.5),
            rnd_effect,
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(blur_limit=5, p=0.05),
            A.Resize(height=height, width=width, p=1),
            rnd_cutout,
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0.3,
            label_fields=['labels']
        )
    )


def get_valid_transforms(height=1024, width=1024):
    """Resize and other transformations for the validation set"""
    return A.Compose(
        [
            A.Resize(height=height, width=width, p=1),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_test_transforms(height=1024, width=1024):
    """Resize and other transformations for doing predictions"""
    return A.Compose([A.Resize(height=height, width=width, p=1), ToTensorV2(p=1)], p=1)