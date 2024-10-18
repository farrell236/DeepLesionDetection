import os
import cv2
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from torchvision import datasets, models, tv_tensors
from torchvision.transforms import v2 as T
from transformers import AutoModelForMaskedLM, AutoTokenizer


class DeepLesion(datasets.CocoDetection):

    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def clip_and_normalize(self,
                           np_image: np.ndarray,
                           clip_min: int = -150,
                           clip_max: int = 250
                           ) -> np.ndarray:
        np_image = np.clip(np_image, clip_min, clip_max)
        np_image = (np_image - clip_min) / (clip_max - clip_min)
        return np_image

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        DICOM_windows = self.coco.loadImgs(id)[0]["windows"]

        # Load Image
        image = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_UNCHANGED)
        image = image.astype('int32') - 32768
        image = self.clip_and_normalize(image, *DICOM_windows)
        image = (image * 255).astype('uint8')
        image = Image.fromarray(np.stack([image]*3, axis=-1), 'RGB')
        image = tv_tensors.Image(image)
        return image

    def _load_target(self, id: int):
        annotations = super()._load_target(id)
        h = self.coco.loadImgs(id)[0]['height']
        w = self.coco.loadImgs(id)[0]['width']

        # Group annotations into a single target dictionary
        # One image can have >1 annotations
        target = {
            'image_id': annotations[0]['image_id'],
            'boxes': T.functional.convert_bounding_box_format(
                tv_tensors.BoundingBoxes(
                    [ann['bbox'] for ann in annotations],
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    canvas_size=(h, w),
                ),
                new_format=tv_tensors.BoundingBoxFormat.XYXY,
            ),
            'labels': torch.tensor([ann['category_id'] for ann in annotations])
        }

        # Check if any annotation contains a 'caption' key
        captions = [ann['caption'] for ann in annotations if 'caption' in ann]
        if captions:
            # Join all captions into one sentence in random order
            caption_text = ' [SEP] '.join(random.sample(captions, len(captions)))
            target['caption'] = self.tokenizer(
                caption_text,
                return_tensors="pt",
                padding="max_length",
                max_length=256
            )

        return target


def get_transform():
    return T.Compose([
        T.ToImage(),
        T.Resize((512, 512)),
        T.ToDtype(torch.float, scale=True),
    ])


if __name__ == '__main__':

    # IMAGES_PATH = '/data/houbb/data/DeepLesion/Images_png'
    # ANNOTATIONS_TRAIN = 'data/cococaption_train_deeplesion.json'
    # ANNOTATIONS_VAL = 'data/cococaption_val_deeplesion.json'

    IMAGES_PATH = '/data/houbb/data/DeepLesion/Images_png'
    ANNOTATIONS_TRAIN = '/data/houbb/data/DeepLesion/annotation/deeplesion_train.json'
    ANNOTATIONS_VAL = '/data/houbb/data/DeepLesion/annotation/deeplesion_val.json'

    dataset = DeepLesion(root=IMAGES_PATH, annFile=ANNOTATIONS_TRAIN, transforms=get_transform())
    dataset_val = DeepLesion(root=IMAGES_PATH, annFile=ANNOTATIONS_VAL, transforms=get_transform())

    a=1

    from helpers import plot
    sample1 = dataset.__getitem__(42)
    sample2 = dataset.__getitem__(102)
    plot((sample1, sample2))

    a=1
