import tensorflow as tf

from model import get_backbone, RetinaNet, DecodePredictions
from deeplesion import test_dataset
from util import preprocess_data3, visualize_detections


a=1

# Load RetinaNet Model
model = RetinaNet(num_classes=1, backbone=get_backbone())
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir="./checkpoints_3"))

# Create Inference Model
image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.3)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

a=1

def plot_sample(sample):

    # Load image and make predictions
    image, bbox, class_id = preprocess_data3(sample)
    detections = inference_model.predict(image[None, ...], verbose=False)
    num_detections = detections.valid_detections[0]

    # Plot Ground Truth
    visualize_detections(
        tf.cast(image, tf.uint8),
        bbox,
        class_id,
        [0]*len(class_id),
        name=f'True: {sample["image_id"].numpy().decode("utf-8")}'
    )

    # Plot Prediction
    visualize_detections(
        tf.cast(image, tf.uint8),
        detections.nmsed_boxes[0][:num_detections],
        detections.nmsed_classes[0][:num_detections],
        detections.nmsed_scores[0][:num_detections],
        name=f'Pred: {sample["image_id"].numpy().decode("utf-8")}'
    )

    a=1

iterator = iter(test_dataset.shuffle(1000))
for i in range(1000):
    sample = next(iterator)
    plot_sample(sample)

a=1

