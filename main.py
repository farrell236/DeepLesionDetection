import os

import cv2
import numpy as np
import torch
import torch.utils.data

from PIL import Image
from torchvision import datasets, models, tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

import utils

from engine import train_one_epoch, evaluate


torch.manual_seed(0)


class DeepLesion(datasets.CocoDetection):

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
        return image


def get_transform():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float, scale=True),
    ])


IMAGES_PATH = '/data/houbb/data/DeepLesion/Images_png'
ANNOTATIONS_TRAIN = '/data/houbb/data/DeepLesion/annotation/deeplesion_train.json'
ANNOTATIONS_VAL = '/data/houbb/data/DeepLesion/annotation/deeplesion_val.json'

dataset = DeepLesion(IMAGES_PATH, ANNOTATIONS_TRAIN, transforms=get_transform())
dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("image_id", "boxes", "labels"))

dataset_val = DeepLesion(IMAGES_PATH, ANNOTATIONS_VAL, transforms=get_transform())
dataset_val = datasets.wrap_dataset_for_transforms_v2(dataset_val, target_keys=("image_id", "boxes", "labels"))


a=1


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# Start training
num_epochs = 32
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    log_stats = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # save model
    torch.save(model.state_dict(), f'checkpoints/fasterrcnn_resnet50_fpn_{epoch}.pth')
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)


a=1
