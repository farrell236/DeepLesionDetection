import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
import numpy as np
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from transformers import AutoTokenizer, AutoModelForMaskedLM

from open_clip import create_model_from_pretrained, get_tokenizer


########################################################################################################################
# CustomFasterRCNN Model
########################################################################################################################

class CustomFasterRCNN(FasterRCNN):
    # inheriting fasterrcnn class to incorporate different backbone
    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#modifying-the-model-to-add-a-different-backbone

    def __init__(self, num_classes=2):
        # load a pre-trained model for classification and return
        # only the features
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        # ``FasterRCNN`` needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        backbone.out_channels = 1280

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        # Initialize the FasterRCNN parent model
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )


########################################################################################################################
# JointLearning_v1 Model
########################################################################################################################

# Convert 1D features to 3D features
# ViT Feature output is [1x512], need to output [128x8x8] for RPN
# note: not restricted to [128x8x8] ¯\_(ツ)_/¯ why not...?
class FeatureTransformer(nn.Module):
    def __init__(self, visual_encoder):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.fc = nn.Linear(512, 128 * 8 * 8)

    def forward(self, pixel_values):
        outputs = self.visual_encoder(pixel_values)
        return self.fc(outputs).view(-1, 128, 8, 8)


class JointLearning_v1(nn.Module):
    # inheriting fasterrcnn class to incorporate CLIP model
    # https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py

    def __init__(self, num_classes=2):
        super(JointLearning_v1, self).__init__()
        clip_model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        backbone = FeatureTransformer(clip_model.visual)
        backbone.out_channels = 128

        # keep same functionality as here because it works
        # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#modifying-the-model-to-add-a-different-backbone
        # Define the anchor generator with specified sizes and aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # Define the ROI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],  # Feature map names expected by the backbone
            output_size=7,  # i think this needs to change to 8??
            sampling_ratio=2
        )

        # Initialize the FasterRCNN parent model
        self.detection_model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=224,  # Pretrained CLIP input is restricted to 224x224
            max_size=224,  # TODO: find transforms and remove augmentation
        )

        self.clip_model = clip_model
        self.clip_loss = nn.CrossEntropyLoss()
        self.dummy_param = nn.Parameter(torch.empty(0))  # Used for model device checking

    def forward(self, images, targets=None):
        loss_dict = self.detection_model(images, targets)

        if targets is not None:
            # Collate images and targets for clip
            images_collated = torch.stack(images, dim=0)
            text_collated = {}
            keys = targets[0]['caption'].keys()
            for key in keys:
                text_collated[key] = torch.cat([d['caption'][key] for d in targets], dim=0).to(self.dummy_param.device)

            image_features, text_features, logit_scale = self.clip_model(images_collated, text_collated)
            logits = logit_scale * image_features @ text_features.t()
            labels = torch.arange(len(images_collated)).to(self.dummy_param.device)
            clip_loss = self.clip_loss(logits, labels)

            loss_dict.update({'loss_clip': clip_loss})

        return loss_dict


########################################################################################################################
# JointLearning_v2 Model
########################################################################################################################

class TextEncoder(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            output_hidden_states=True)
        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, texts):
        x = self.model(input_ids=texts['input_ids'], attention_mask=texts['attention_mask'])['hidden_states'][-1]
        x = x[:, 0, :]  # B, T[cls], E
        x = self.projection(x)
        x = self.layer_norm(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, base_model, embed_dim, proj_dim):
        super().__init__()

        self.model = nn.Sequential(
            base_model,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, x):
        return self.model(x)


class CLIPModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.image_encoder = ImageEncoder(base_model=base_model, embed_dim=1280, proj_dim=256)
        self.text_encoder = TextEncoder(embed_dim=768, proj_dim=256)

        self.temperature = nn.Parameter(torch.ones([])*np.log(1/0.07))

    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)

        return image_features, text_features, self.temperature.exp()


class JointLearning_v2(nn.Module):

    def __init__(self, num_classes=2):
        super(JointLearning_v2, self).__init__()

        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        backbone.out_channels = 1280

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        self.detection_model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        # state_dict = torch.load('jointmodel/jointmodel_9.pth')
        # self.detection_model.load_state_dict(state_dict)
        # for param in self.detection_model.parameters():
        #     param.requires_grad = False

        self.clip_model = CLIPModel(backbone)
        self.clip_loss = nn.CrossEntropyLoss()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, images, targets=None):
        loss_dict = self.detection_model(images, targets)

        if targets is not None:
            # Collate images and targets for clip
            images_collated = torch.stack(images, dim=0)
            text_collated = {}
            keys = targets[0]['caption'].keys()
            for key in keys:
                text_collated[key] = torch.cat([d['caption'][key] for d in targets], dim=0).to(self.dummy_param.device)

            image_features, text_features, logit_scale = self.clip_model(images_collated, text_collated)
            logits = logit_scale * image_features @ text_features.t()
            labels = torch.arange(len(images_collated)).to(self.dummy_param.device)
            clip_loss = self.clip_loss(logits, labels)

            loss_dict.update({'loss_clip': clip_loss})

        return loss_dict


if __name__ == '__main__':


    from deeplesion import DeepLesion, get_transform
    import utils

    IMAGES_PATH = '/data/houbb/data/DeepLesion/Images_png'
    ANNOTATIONS_TRAIN = 'data/cococaption_train_deeplesion.json'
    ANNOTATIONS_VAL = 'data/cococaption_val_deeplesion.json'

    dataset = DeepLesion(root=IMAGES_PATH, annFile=ANNOTATIONS_TRAIN, transforms=get_transform())

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        collate_fn=utils.collate_fn
    )

    image, targets = next(iter(data_loader))

    model = JointLearning_v1(num_classes=2)
    out = model(image, targets)

    a=1
