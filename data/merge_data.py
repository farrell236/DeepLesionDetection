import json

import numpy as np
import pandas as pd


a=1

train_df = pd.read_json('train_deeplesion.json')
val_df = pd.read_json('val_deeplesion.json')
test_df = pd.read_json('test_deeplesion.json')

df = pd.concat([train_df, val_df, test_df]).set_index('lesion_idx').sort_index()

dl_info = pd.read_csv('DL_info.csv')
dl_info = dl_info.loc[df.index]


def calculate_ellipse_area(row):
    axes = row.split(',')
    semi_major_axis = float(axes[0].strip()) / 2
    semi_minor_axis = float(axes[1].strip()) / 2
    return np.pi * semi_major_axis * semi_minor_axis


df_new = pd.DataFrame()
df_new['file_name'] = dl_info['File_name'].apply(lambda x: '/'.join(x.rsplit('_', 1)))
df_new[['height', 'width']] = dl_info['Image_size'].str.split(', ', expand=True).astype(int)
df_new['slice_no'] = dl_info['Key_slice_index']
df_new['spacing'] = dl_info['Spacing_mm_px_'].str.split(', ', expand=True)[1].astype(float)
df_new['slice_intv'] = dl_info['Spacing_mm_px_'].str.split(', ', expand=True)[2].astype(float)
df_new['z_position'] = dl_info['Normalized_lesion_location'].str.split(', ', expand=True)[2].astype(float)
df_new['windows'] = dl_info['DICOM_windows'].apply(lambda x: [float(num) for num in x.split(', ')])
df_new['area'] = dl_info['Lesion_diameters_Pixel_'].apply(calculate_ellipse_area)
df_new['bbox'] = dl_info['Bounding_boxes'].apply(lambda x: [float(num) for num in x.split(', ')])
df_new['noisy'] = dl_info['Possibly_noisy']
df_new['id'], _ = pd.factorize(df_new['file_name'])  # this should be unique image id

images_df = df_new[['file_name', 'height', 'width', 'id', 'slice_no', 'spacing', 'slice_intv', 'z_position', 'windows']]
images_df = images_df.drop_duplicates(subset='file_name', keep='first')

annotations_df = df_new[['area', 'id', 'bbox', 'noisy']]
annotations_df = annotations_df.rename(columns={'id': 'image_id'})
annotations_df['id'] = annotations_df.index  # using as lesion id as unique annotation id
annotations_df['iscrowd'] = 0
annotations_df['category_id'] = 1
annotations_df['segmentation'] = annotations_df.apply(lambda x: [[]], axis=1)
annotations_df['caption'] = df.loc[annotations_df.index]['description']
annotations_df = annotations_df[['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id', 'noisy', 'caption']]


coco_info = {
    'description': 'NIH DeepLesion Dataset.',
    'url': 'https://doi.org/10.1117/1.JMI.5.3.036501',
    'version': 'v1.0',
    'year': 2018,
    'contributor': 'NIH',
    'date_created': '2019.01.10'
}

coco_licenses = {
    'url': 'None',
    'id': 1,
    'name': 'All rights reserved by xxx'
}

coco_categories = {
    'supercategory': 'DeepLesion',
    'id': 1,
    'name': 'Lesion'
}

a=1

train_out = {
    'info': coco_info,
    'licenses': coco_licenses,
    'images': images_df[images_df.index.isin(train_df['lesion_idx'])].to_dict(orient='records'),
    'categories': [coco_categories],
    'annotations': annotations_df[annotations_df.index.isin(train_df['lesion_idx'])].to_dict(orient='records'),
}

with open('cococaption_train_deeplesion.json', 'w') as file:
    json.dump(train_out, file, indent=4)


val_out = {
    'info': coco_info,
    'licenses': coco_licenses,
    'images': images_df[images_df.index.isin(val_df['lesion_idx'])].to_dict(orient='records'),
    'categories': [coco_categories],
    'annotations': annotations_df[annotations_df.index.isin(val_df['lesion_idx'])].to_dict(orient='records'),
}

with open('cococaption_val_deeplesion.json', 'w') as file:
    json.dump(val_out, file, indent=4)


test_out = {
    'info': coco_info,
    'licenses': coco_licenses,
    'images': images_df[images_df.index.isin(test_df['lesion_idx'])].to_dict(orient='records'),
    'categories': [coco_categories],
    'annotations': annotations_df[annotations_df.index.isin(test_df['lesion_idx'])].to_dict(orient='records'),
}

with open('cococaption_test_deeplesion.json', 'w') as file:
    json.dump(test_out, file, indent=4)

