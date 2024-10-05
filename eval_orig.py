import os

import tensorflow as tf

from model import LabelEncoder, RetinaNet, RetinaNetLoss, get_backbone, DecodePredictions
# from vinbigdata import train_dataset, val_dataset
from deeplesion import test_dataset
from util import preprocess_data2, preprocess_data3, resize_and_pad_image, visualize_detections, CLASSES, tf_clip_and_normalize

import matplotlib.pyplot as plt

a=1


model_dir = "./checkpoints"
label_encoder = LabelEncoder()

num_classes = 1 # 15
batch_size = 2

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)


a=1


latest_checkpoint = tf.train.latest_checkpoint(model_dir)
model.load_weights(latest_checkpoint)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    # image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

def plot_sample(sample):
    image = tf.io.decode_png(tf.io.read_file(sample["image"]), channels=3,
                         dtype=tf.dtypes.uint16)
    image = tf.cast(image, tf.int32) - 2 ** 15
    image = tf.cast(tf_clip_and_normalize(image) * 255, tf.int32)
    # image = tf.io.decode_jpeg(tf.io.read_file(sample["image"]), channels=3)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        f'{CLASSES[int(x)]}' for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )


iterator = iter(test_dataset.shuffle(1000))

a=1

sample = next(iterator)
i, b, l, ratio = preprocess_data3(sample)
l = [CLASSES[int(x.numpy())] for x in l]
visualize_detections(i.numpy().astype('int32'), b, l, [0]*len(l))
plot_sample(sample)
