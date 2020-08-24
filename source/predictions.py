"""
Module that implements functions for doing inference, including:
- Test-Time Augmentation (TTA) for creating multiple predictions
- Weighted Box Fusion (WBF) to combine predictions
"""

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import ensemble_boxes

class BaseTTA:
    """Base class for Test-Time Augmentation transformations"""
    def __init__(self, image_size=512):
        self.image_size = image_size

    def augment(self, image):
        raise NotImplementedError
    
    def batch_augment(self, images):
        raise NotImplementedError
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

        
class TTAHorizontalFlip(BaseTTA):
    def augment(self, image):
        return image.flip(1)
    
    def batch_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return boxes

    
class TTAVerticalFlip(BaseTTA):
    def augment(self, image):
        return image.flip(2)
    
    def batch_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes
    
    
class TTARotate90(BaseTTA):
    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return res_boxes
    
    
class TTARotate180(TTARotate90): # apply rotate90 twice
    def augment(self, image):
        return super().augment(super().augment(image))

    def batch_augment(self, images):
        return super().batch_augment(super().batch_augment(images))
    
    def deaugment_boxes(self, boxes):
        return super().deaugment_boxes(super().deaugment_boxes(boxes))
    

class TTASaturation(BaseTTA):
    transform = A.HueSaturationValue(hue_shift_limit=0.2,
                                     sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2,p=1)
    
    def augment(self,image):
        image = image.permute(1,2,0).cpu().numpy()
        image = self.transform(image=image)['image']
        return ToTensorV2()(image=image)['image']

    def batch_augment(self, images):
        batch_size = len(images)
        for i in range(batch_size):
            images[i] = self.augment(images[i])
        return images
        
    def deaugment_boxes(self,boxes):
        return boxes


class TTARBC(BaseTTA):
    transform = A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.5)
    
    def augment(self, image):
        image = image.permute(1,2,0).cpu().numpy()
        image = self.transform(image=image)['image']
        return ToTensorV2()(image=image)['image']

    def batch_augment(self, images):
        batch_size = len(images)
        for i in range(batch_size):
            images[i] = self.augment(images[i])
        return images
        
    def deaugment_boxes(self,boxes):
        return boxes


class TTABlur(BaseTTA):
    transform = A.Blur(blur_limit=5, p=1)
    
    def augment(self, image):
        image = image.permute(1,2,0).cpu().numpy()
        image = self.transform(image=image)['image']
        return ToTensorV2()(image=image)['image']

    def batch_augment(self, images):
        batch_size = len(images)
        for i in range(batch_size):
            images[i] = self.augment(images[i])
        return images
        
    def deaugment_boxes(self,boxes):
        return boxes


class TTACompose(BaseTTA):
    def __init__(self, transforms):
        self.transforms = transforms

    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image
    
    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images
    
    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)


def process_det(index, det, score_threshold=0.25):
    """Function for post-processing detections from pytorch"""
    boxes = det[index].detach().cpu().numpy()[:,:4]    
    scores = det[index].detach().cpu().numpy()[:,4]
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
    boxes = (boxes).clip(min=0, max=511).astype(int)
    indexes = np.where(scores>score_threshold)
    boxes = boxes[indexes]
    scores = scores[indexes]
    return boxes, scores


def make_tta_predictions(images, tta_transforms, net, score_threshold=0.25):
    """Make predictions for each TTA transformation using our model"""
    with torch.no_grad():
        images = torch.stack(images).float().cuda()
        predictions = []
        for tta_transform in tta_transforms:
            result = []
            det = net(tta_transform.batch_augment(images.clone()), torch.tensor([1]*images.shape[0]).float().cuda())

            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:,:4]    
                scores = det[i].detach().cpu().numpy()[:,4]
                indexes = np.where(scores > score_threshold)[0]
                boxes = boxes[indexes]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                boxes = tta_transform.deaugment_boxes(boxes.copy())
                result.append({
                    'boxes': boxes,
                    'scores': scores[indexes],
                })
            predictions.append(result)
    return predictions


def run_wbf(predictions, image_index, image_size=512,
            iou_thr=0.44,skip_box_thr=0.43, weights=None):
    """Run Weighted Box-Fusion to combine many predicted bounding boxes (see references)."""
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()
             for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()
              for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist()
              for prediction in predictions]
    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(
        boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels