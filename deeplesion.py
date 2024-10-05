import cv2
import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

import pandas as pd

from glob import glob
from tqdm import tqdm

from sklearn.model_selection import StratifiedGroupKFold


def clip_and_normalize(np_image: np.ndarray,
                       clip_min: int = -125,
                       clip_max: int = 225
                       ) -> np.ndarray:
    np_image = np.clip(np_image, clip_min, clip_max)
    np_image = (np_image - clip_min) / (clip_max - clip_min)
    return np_image


def tf_clip_and_normalize(tf_image, clip_min, clip_max):
    tf_image = tf.keras.backend.clip(tf_image, clip_min, clip_max)
    tf_image = (tf_image - clip_min) / (clip_max - clip_min)
    return tf_image

a=1




data_root = '/data/houbb/data/DeepLesion'

dataset_df = pd.read_csv('/data/houbb/data/DeepLesion/DL_info.csv')
# dataset_df = dataset_df[dataset_df['Possibly_noisy'] == 0]

##########
df = pd.DataFrame()

df['File_name'] = dataset_df['File_name'].apply(lambda x: '/'.join(x.rsplit('_', 1)))
df['Coarse_lesion_type'] = dataset_df['Coarse_lesion_type']
# df['Coarse_lesion_type'] = df['Coarse_lesion_type'].replace(-1, 0)
df[['x_min', 'y_min', 'x_max', 'y_max']] = dataset_df['Bounding_boxes'].str.split(', ', expand=True).astype(float)
df[['height', 'width']] = dataset_df['Image_size'].str.split(', ', expand=True).astype(int)
df['Train_Val_Test'] = dataset_df['Train_Val_Test']
# df = df[df['Coarse_lesion_type'] == -1]

a=1

##########

train_df = df[df['Train_Val_Test'] == 1]
val_df = df[df['Train_Val_Test'] == 2]
test_df = df[df['Train_Val_Test'] == 3]


a=1


# s = train_df.iloc[42]
#
# t = tf.io.decode_png(tf.io.read_file(os.path.join(data_root, 'Images_png', s['File_name'])), channels=3, dtype=tf.dtypes.uint16)
# t = tf.cast(t, tf.int32) - 2 ** 15
#
#
#
# image = cv2.imread(os.path.join(data_root, 'Images_png', s['File_name']), -1).astype('int32') - 2**15
# image = clip_and_normalize(image)
#
# plt.imshow(image, cmap='gray')
# plt.show()
#
#
# image2 = cv2.rectangle(image, (int(s['x_min']), int(s['y_min'])), (int(s['x_max']), int(s['y_max'])), 1, 2)
#
# plt.imshow(image2, cmap='gray')
# plt.show()


# for idx, row in tqdm(df.iterrows(), total=len(df)):
#     if not os.path.exists(os.path.join(data_root, 'Images_png', row['File_name'])):
#         print(idx, row['File_name'])


def scale_images(df):
    # scale the coordinates of the bounding boxes from their initial values to fit the 512x512 images
    # set to the images with no object (class 14), bounding box with coordinates [xmin=0 ymin=0 xmax=1 ymax=1]
    # train_df.loc[train_df["class_id"] == 14, ['x_max', 'y_max']] = 1.0
    # train_df.loc[train_df["class_id"] == 14, ['x_min', 'y_min']] = 0
    copy_df = df.copy()
    # scale the input image coordinates to fit 512x512 image
    copy_df['xmin'] = (df['x_min']/df['width'])
    copy_df['ymin'] = (df['y_min']/df['height'])
    copy_df['xmax'] = (df['x_max']/df['width'])
    copy_df['ymax'] = (df['y_max']/df['height'])

    # train_df.loc[train_df["class_id"] == 14, ['xmax', 'ymax']] = 1.0
    # train_df.loc[train_df["class_id"] == 14, ['xmin', 'ymin']] = 0

    return copy_df

a=1



train_df = scale_images(train_df)
val_df = scale_images(val_df)
test_df = scale_images(test_df)

def meta_dict_gen(dataset_df):
    image_ids = dataset_df['File_name'].unique()
    def _meta_dict_gen():
        for image_id in image_ids:
            sample = dataset_df[dataset_df['File_name'] == image_id]
            image = data_root + '/Images_png/' + sample['File_name'].values[0]
            bbox = sample[['xmin', 'ymin', 'xmax', 'ymax']].values
            class_id = sample['Coarse_lesion_type'].values * 0
            yield {'image': image, 'bbox': bbox, 'label': class_id, 'image_id': image_id}
    return _meta_dict_gen

train_dataset = tf.data.Dataset.from_generator(
    meta_dict_gen(train_df),
    output_types={'image': tf.string, 'bbox': tf.float32, 'label': tf.int32, 'image_id': tf.string})

val_dataset = tf.data.Dataset.from_generator(
    meta_dict_gen(val_df),
    output_types={'image': tf.string, 'bbox': tf.float32, 'label': tf.int32, 'image_id': tf.string})

test_dataset = tf.data.Dataset.from_generator(
    meta_dict_gen(test_df),
    output_types={'image': tf.string, 'bbox': tf.float32, 'label': tf.int32, 'image_id': tf.string})


a=1

# iterator = iter(train_dataset)
# batch = next(iterator)


a=1

# dl_iterator = iter(test_dataset)
# dl_batch = next(dl_iterator)