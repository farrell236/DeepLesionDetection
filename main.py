import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

from model import LabelEncoder, RetinaNet, RetinaNetLoss, get_backbone
# from vinbigdata import train_dataset, val_dataset
from deeplesion import train_dataset, val_dataset
from util import preprocess_data2

import wandb
from wandb.integration.keras import WandbMetricsLogger

# Ensure the W&B API key is set
if not os.getenv('WANDB_API_KEY'):
    raise ValueError("WANDB_API_KEY environment variable not set.")

wandb.login()

# Initialize a W&B run with the specified project, entity, group, and config
wandb.init(project='RetinaNet', entity="farrell236", name=f'retinanet_3')


a=1


checkpoint_dir = './checkpoints_3'
label_encoder = LabelEncoder()

num_classes = 1  # 15
batch_size = 16
epochs = 300

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
# learning_rate_boundaries = [125*2, 250*4, 500*8, 240000*16, 360000*32]
learning_rate_boundaries = [10, 20, 40, 80, 100]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate_fn, beta_1=0.9, beta_2=0.98)
model.compile(loss=loss_fn, optimizer=optimizer)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "weights" + "_epoch_{epoch}"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    WandbMetricsLogger(),
]


autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data2, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(8 * batch_size)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padded_shapes=([None,None,3],[None,4],[None]), padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(preprocess_data2, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padded_shapes=([None,None,3],[None,4],[None]), padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)


a=1

iterator = iter(train_dataset)
batch = next(iterator)
#
model(batch[0])
model.load_weights('./checkpoints_2/weights_epoch_91')

a=1

# Uncomment the following lines, when training on full dataset
# train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
# val_steps_per_epoch = \
#     dataset_info.splits["validation"].num_examples // batch_size

# train_steps = 4 * 100000
# epochs = train_steps // train_steps_per_epoch


# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    initial_epoch=100,
    callbacks=callbacks_list,
    verbose=1,
)

a=1
