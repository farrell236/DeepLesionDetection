import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import nn, Tensor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from open_clip import create_model_from_pretrained, get_tokenizer


# Convert 1D features to 3D features
# ViT Feature output is [1x512], need to output [128x8x8] for RPN
# note: not restricted to [128x8x8] ¯\_(ツ)_/¯ why not...?
class FeatureTransformer(torch.nn.Module):
    def __init__(self, visual_encoder):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.fc = nn.Linear(512, 128 * 8 * 8)

    def forward(self, pixel_values):
        outputs = self.visual_encoder(pixel_values)
        return self.fc(outputs).view(-1, 128, 8, 8)


class JointLearning(FasterRCNN):
    # inheriting fasterrcnn class to incorporate CLIP model
    # https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py

    def __init__(self, num_classes=2):
        # Load a CLIP model and extract the features
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
            featmap_names=['0'],  # Feature map names expected by the backbone
            output_size=7,  # i think this needs to change to 8??
            sampling_ratio=2
        )

        # Initialize the FasterRCNN parent model
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=224,  # CLIP input is restricted to 224x224
            max_size=224,  # TODO: find transforms and remove augmentation
        )

        self.clip_model = clip_model
        self.clip_criterion = nn.CrossEntropyLoss()
        self.dummy_param = nn.Parameter(torch.empty(0))  # Used for model device checking

    # Override GeneralizedRCNN forward function to include joint learning loss
    # Code is largely borrowed from:
    # https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py
    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        # Adding Forward method for CLIP loss
        # note: .to(self.dummy_param.device) here is a bit of a hack... but works...
        if targets is not None:
            tmp = torch.stack([t["caption"][0] for t in targets]).to(self.dummy_param.device)  # taking the first caption only... TODO: fixme!
            image_features, text_features, logit_scale = self.clip_model(images.tensors, tmp)
            logits = logit_scale * image_features @ text_features.t()
            labels = torch.arange(len(images.tensors)).to(self.dummy_param.device)
            clip_losses = self.clip_criterion(logits, labels)
        else:
            clip_losses = 0

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update({'loss_clip': clip_losses})

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


if __name__ == '__main__':

    a=1
