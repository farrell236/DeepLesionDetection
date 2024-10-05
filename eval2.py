import pandas as pd
import numpy as np
import tensorflow as tf

from model import get_backbone, RetinaNet, DecodePredictions
from deeplesion import test_dataset, test_df
from util import preprocess_data3, visualize_detections

from tqdm import tqdm

a=1

# Load RetinaNet Model
model = RetinaNet(num_classes=1, backbone=get_backbone())
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir="./checkpoints_3"))

# Create Inference Model
image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.4)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

a=1


def calculate_iou(pred_box, gt_box):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Bounding boxes are represented as [x_min, y_min, x_max, y_max].
    """
    # Determine the coordinates of the intersection rectangle
    x_min_inter = max(pred_box[0], gt_box[0])
    y_min_inter = max(pred_box[1], gt_box[1])
    x_max_inter = min(pred_box[2], gt_box[2])
    y_max_inter = min(pred_box[3], gt_box[3])

    # Compute the area of intersection
    inter_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    pred_box_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    gt_box_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)

    # Compute the area of union
    union_area = pred_box_area + gt_box_area - inter_area

    # Return IoU
    iou = inter_area / union_area
    return iou


def calculate_false_positives_at_iou_levels(pred_boxes, gt_boxes, iou_levels=[0.5, 1.0, 2.0, 3.0, 4.0]):
    """
    Calculate false positives per image at different IoU threshold levels.
    pred_boxes: list of predicted bounding boxes, each in [x_min, y_min, x_max, y_max] format
    gt_boxes: list of ground truth bounding boxes, each in [x_min, y_min, x_max, y_max] format
    iou_levels: list of IoU thresholds at which to calculate false positives
    """
    false_positives_at_levels = {}

    for iou_threshold in iou_levels:
        false_positives = 0
        matched_gt = [False] * len(gt_boxes)

        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            # Find the best matching ground truth box with the highest IoU
            for i, gt_box in enumerate(gt_boxes):
                if not matched_gt[i]:  # Only consider unmatched ground truth boxes
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

            # Count false positives if IoU is below the current threshold
            if best_iou < iou_threshold:
                false_positives += 1

        false_positives_at_levels[f"IoU {iou_threshold}"] = false_positives

    return false_positives_at_levels





def evaluate_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Evaluate object detection predictions based on IoU.
    pred_boxes: list of predicted bounding boxes, each in [x_min, y_min, x_max, y_max] format
    gt_boxes: list of ground truth bounding boxes, each in [x_min, y_min, x_max, y_max] format
    iou_threshold: IoU threshold to consider a detection as true positive
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Track ground truth box matches
    matched_gt = [False] * len(gt_boxes)

    # Iterate over each predicted box
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        # Find the ground truth box with the highest IoU
        for i, gt_box in enumerate(gt_boxes):
            if not matched_gt[i]:  # Only consider unmatched ground truth boxes
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

        # Determine if this is a true positive or false positive
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt[best_gt_idx] = True
        else:
            false_positives += 1

    # Any unmatched ground truth box is a false negative
    false_negatives = matched_gt.count(False)

    return {
        "True Positives": true_positives,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
        "Precision": true_positives / (true_positives + false_positives) if (
                                                                                        true_positives + false_positives) > 0 else 0,
        "Recall": true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0,
        "F1 Score": (2 * true_positives) / (2 * true_positives + false_positives + false_negatives) if (
                    true_positives > 0) else 0
    }

results = []
def plot_sample(sample):

    # Load image and make predictions
    image, bbox, class_id = preprocess_data3(sample)
    detections = inference_model.predict(image[None, ...], verbose=False)
    num_detections = detections.valid_detections[0]

    result = evaluate_predictions(
        pred_boxes=detections.nmsed_boxes[0][:num_detections],
        gt_boxes=bbox)
    # result = calculate_false_positives_at_iou_levels(
    #     pred_boxes=detections.nmsed_boxes[0][:num_detections],
    #     gt_boxes=bbox)

    results.append({
        'image_id': sample['image_id'],
        **result
    })

    a=1



for sample in tqdm(test_dataset, total=len(test_df)):
    plot_sample(sample)


results_df = pd.DataFrame(results)
results_df.to_csv('results_fp.csv', index=False)

a=1

