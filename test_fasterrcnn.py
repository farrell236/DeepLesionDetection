import os

import cv2
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt

from torchvision import datasets, models, tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes

import utils

from engine import evaluate

from deeplesion import DeepLesion, get_transform


torch.manual_seed(0)


IMAGES_PATH = '/data/houbb/data/DeepLesion/Images_png'
# ANNOTATIONS_TEST = 'data/cococaption_test_deeplesion.json'
# ANNOTATIONS_TRAIN = '/data/houbb/data/DeepLesion/annotation/deeplesion_train.json'
# ANNOTATIONS_VAL = '/data/houbb/data/DeepLesion/annotation/deeplesion_val.json'
ANNOTATIONS_TEST = '/data/houbb/data/DeepLesion/annotation/deeplesion_test.json'

dataset_test = DeepLesion(IMAGES_PATH, ANNOTATIONS_TEST, transforms=get_transform())


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

# define training and validation data loaders
# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test,
#     batch_size=1,
#     shuffle=False,
#     collate_fn=utils.collate_fn
# )

# get the model using our helper function
model = get_model_instance_segmentation(num_classes=2)

# Load the state dictionary from the checkpoint
state_dict = torch.load('fasterrcnn_resnet50_fpn.pth')

# Load the state dictionary into the model
model.load_state_dict(state_dict)
model.to(device)
model.eval()


# result = evaluate(model, data_loader_test, device=device)

idx = np.random.randint(42, 128, 20)
for i in idx:
    with torch.no_grad():
        image, target = dataset_test.__getitem__(i)
        # convert RGBA -> RGB and move to device
        image = image.to(device)
        predictions = model([image, ])
        pred = predictions[0]

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    pred_boxes = pred["boxes"].long()
    true_boxes = target['boxes'].long()
    pred_image = draw_bounding_boxes(image, pred_boxes[:3,:], colors="red").cpu()
    true_image = draw_bounding_boxes(image, true_boxes, colors="blue").cpu()

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot predicted bounding boxes
    axs[0].imshow(pred_image.permute(1, 2, 0))
    axs[0].set_title('Predicted Boxes')
    axs[0].axis('off')  # Hide axis

    # Plot ground truth bounding boxes
    axs[1].imshow(true_image.permute(1, 2, 0))
    axs[1].set_title('Ground Truth Boxes')
    axs[1].axis('off')  # Hide axis

    # Display the plots
    # plt.tight_layout()
    plt.savefig(f'imgdump_2/sample_{i}.png')
    plt.show()

a=1

# from tqdm import tqdm
#
# for image, target in tqdm(dataset_test):
#     with torch.no_grad():
#         image = image.to(device)
#         predictions = model([image, ])
#         pred = predictions[0]
#
#     image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
#     pred_boxes = pred["boxes"].long()
#     true_boxes = target['boxes'].long()
#     pred_image = draw_bounding_boxes(image, pred_boxes[:1,:], colors=(0,255,0), width=2).cpu()
#
#     filename = target['filename'].replace('/','_')
#     cv2.imwrite(f'pred_images/train/{filename}', pred_image.permute(1, 2, 0).numpy())
#
#     # plt.imshow(pred_image.permute(1, 2, 0).numpy())
#     # plt.show()
