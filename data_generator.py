# -*- coding: utf-8 -*-

import imageio
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imsave
import os
import os.path
from scipy.ndimage import median_filter
import shutil
import Amir_utils
import time
import cv2
import pandas as pd
from preprocessing import Filter
import argparse
#%%
data_count = 0
data_all = []
data_names = []

parser = argparse.ArgumentParser(description='data')
parser.add_argument('--filtname', default='median', type=str, help ='preprocessing filter name')
parser.add_argument('--aug', default='', action='store_true', dest='augment',help ='add augmentation')
parser.add_argument('--no_aug', default='', action='store_false', dest='augment',help ='dont add augmentation')
parser.add_argument('--type_patch', default='pad', type=str, help= 'type of patch extraction --> by zero padding or not, by default pad, other opt = no_pad')
parser.set_defaults(augment = True)
parser.add_argument('--thick_mask_lines', default=4, type=int, help='make mask lines thicker by number/iterations')
parser.add_argument('--kernel_size', default=3, type=int, help='kernel size for line thickening/dilation')
parser.add_argument('--num_augmentations', default=1, type=int, help='number of augmentations')

args = parser.parse_args()
pre_name = args.filtname
augment = args.augment
type_of_patch_extraction = args.type_patch
thick_line = args.thick_mask_lines
size = args.kernel_size
num_augments = args.num_augmentations

print('augmentation = ' + str(augment))

from Utils import remove_images, apply_filter


traincsv_path = 'train_images.csv'
valcsv_path = 'validation_images.csv'
testcsv_path = 'test_images.csv'

#dataframes from csv file
df_train = pd.read_csv(traincsv_path)
#df_train = remove_images(df_train, 5)
#df_train = remove_images(df_train, 6)
train_images_id = df_train.iloc[0:,1]
train_zones_id = df_train.iloc[0:,2]
train_front_id = df_train.iloc[0:,3]

df_val = pd.read_csv(valcsv_path)
#df_val = remove_images(df_val, 5)
#df_val = remove_images(df_val, 6)
val_images_id = df_val.iloc[0:,1]
val_zones_id = df_val.iloc[0:,2]
val_front_id = df_val.iloc[0:,3]

df_test = pd.read_csv(testcsv_path)
#df_test = remove_images(df_test, 5)
#df_test = remove_images(df_test, 6)
test_images_id = df_test.iloc[0:,1]
test_zones_id = df_test.iloc[0:,2]
test_front_id = df_test.iloc[0:,3]


#for i in range(len(train_images_id)):
#    print(train_images_id.iloc[i])

PATCH_SIZE = 256 # ERROR: Some images are smaller than 512 and thus will be discarded with PATCH_SIZE=512 (We do not want it!)

# patch extraction of rightside separately
def patch_rightborder(image, patch_size):
    height = image.shape[0]
    width = image.shape[1]

    total_patches = int((height - height%patch_size)/patch_size)
    all_right_patches = np.zeros((total_patches, patch_size, patch_size))
    for i in range(total_patches):
        all_right_patches[i,:] = image[patch_size*(i):patch_size*(i+1), (width - 1 - patch_size):(width - 1)]
    return all_right_patches

# patch extraction of bottom separately
def patch_bottomborder(image, patch_size):
    height = image.shape[0]
    width = image.shape[1]
    total_patches = int((width - width % patch_size) / patch_size)
    all_bottom_patches = np.zeros((total_patches, patch_size, patch_size))
    for i in range(total_patches):
        all_bottom_patches[i,:] = image[(height - 1 - patch_size):(height - 1), (width-1)-patch_size*(i+1):(width-1)-patch_size*i]
    return all_bottom_patches

#####
# train path
train_img_path = str(Path('data_'+str(PATCH_SIZE)+pre_name+'_linesw'+str(size)+'i'+str(thick_line)+ '/train/images/'))
train_zones_path = str(Path('data_'+str(PATCH_SIZE)+pre_name+'_linesw'+str(size)+'i'+str(thick_line)+ '/train/masks_zones'))
train_lines_path = str(Path('data_'+str(PATCH_SIZE)+pre_name+'_linesw'+str(size)+'i'+str(thick_line)+ '/train/masks_lines'))
if not os.path.exists(train_img_path): os.makedirs(train_img_path)
if not os.path.exists(train_zones_path): os.makedirs(train_zones_path)
if not os.path.exists(train_lines_path): os.makedirs(train_lines_path)

STRIDE_train = (PATCH_SIZE,PATCH_SIZE)
patch_counter_train = 0

print("Creating Train images")

if type_of_patch_extraction is 'pad':

    for i in range(len(train_images_id)):
        train_images_img = imageio.imread(train_images_id.iloc[i])
        train_images_img = apply_filter(pre_name, train_images_img)
        train_zones_img = imageio.imread(train_zones_id.iloc[i])
        train_front_img = imageio.imread(train_front_id.iloc[i])

        #padding
        train_images_img = cv2.copyMakeBorder(train_images_img, 0, (PATCH_SIZE - train_images_img.shape[0]) % PATCH_SIZE, 0,
                                     (PATCH_SIZE - train_images_img.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
        train_zones_img = cv2.copyMakeBorder(train_zones_img, 0, (PATCH_SIZE - train_zones_img.shape[0]) % PATCH_SIZE, 0,
                                     (PATCH_SIZE - train_zones_img.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
        train_front_img = cv2.copyMakeBorder(train_front_img, 0, (PATCH_SIZE - train_front_img.shape[0]) % PATCH_SIZE, 0,
                                     (PATCH_SIZE - train_front_img.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)

        masks_zone_tmp = train_zones_img
        masks_zone_tmp[masks_zone_tmp == 127] = 0
        masks_zone_tmp[masks_zone_tmp == 254] = 255

        # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement

        p_masks_zone, i_masks_zone = Amir_utils.extract_grayscale_patches(masks_zone_tmp, (PATCH_SIZE, PATCH_SIZE),
                                                                          stride=STRIDE_train)
        p_img, i_img = Amir_utils.extract_grayscale_patches(train_images_img, (PATCH_SIZE, PATCH_SIZE),
                                                            stride=STRIDE_train)
        p_masks_line, i_masks_line = Amir_utils.extract_grayscale_patches(train_front_img,
                                                                          (PATCH_SIZE, PATCH_SIZE), stride=STRIDE_train)
        for j in range(p_masks_zone.shape[0]):
            # if np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) > 0.05 and np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) < 0.95: # only those patches that has both background and foreground
            if np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) >= 0 and np.count_nonzero(p_masks_zone[j]) / (
                    PATCH_SIZE * PATCH_SIZE) <= 1:
                cv2.imwrite(
                    str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/train/images/' + str(patch_counter_train) + '.png')),
                    p_img[j])
                cv2.imwrite(str(
                    Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/train/masks_zones/' + str(patch_counter_train) + '.png')),
                            p_masks_zone[j])
                # thicken lines ground truths
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
                p_masks_line[j] = cv2.dilate(p_masks_line[j], kernel, iterations=thick_line)
                cv2.imwrite(str(
                    Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/train/masks_lines/' + str(patch_counter_train) + '.png')),
                            p_masks_line[j])
                patch_counter_train += 1

# no padding case, extract patches from right and bottom separately
else:
    for i in range(len(train_images_id)):
        train_images_img = imageio.imread(train_images_id.iloc[i])
        train_images_img = apply_filter(pre_name, train_images_img)
        train_zones_img = imageio.imread(train_zones_id.iloc[i])
        train_front_img = imageio.imread(train_front_id.iloc[i])


        masks_zone_tmp = train_zones_img
        masks_zone_tmp[masks_zone_tmp == 127] = 0
        masks_zone_tmp[masks_zone_tmp == 254] = 255

        # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement

        p_masks_zone, i_masks_zone = Amir_utils.extract_grayscale_patches(masks_zone_tmp, (PATCH_SIZE, PATCH_SIZE),
                                                                          stride=STRIDE_train)
        p_img, i_img = Amir_utils.extract_grayscale_patches(train_images_img, (PATCH_SIZE, PATCH_SIZE),
                                                            stride=STRIDE_train)
        p_masks_line, i_masks_line = Amir_utils.extract_grayscale_patches(train_front_img,
                                                                          (PATCH_SIZE, PATCH_SIZE), stride=STRIDE_train)
        for j in range(p_masks_zone.shape[0]):
            # if np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) > 0.05 and np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) < 0.95: # only those patches that has both background and foreground
            if np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) >= 0 and np.count_nonzero(p_masks_zone[j]) / (
                    PATCH_SIZE * PATCH_SIZE) <= 1:
                cv2.imwrite(
                    str(Path('data_' + str(PATCH_SIZE) + pre_name+'_linesw'+str(size)+'i'+str(thick_line) + '/train/images/' + str(patch_counter_train) + '.png')),
                    p_img[j])
                cv2.imwrite(str(
                    Path('data_' + str(PATCH_SIZE) + pre_name+'_linesw'+str(size)+'i'+str(thick_line) + '/train/masks_zones/' + str(patch_counter_train) + '.png')),
                            p_masks_zone[j])
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
                p_masks_line[j] = cv2.dilate(p_masks_line[j], kernel, iterations=thick_line)
                cv2.imwrite(str(
                    Path('data_' + str(PATCH_SIZE) + pre_name+'_linesw'+str(size)+'i'+str(thick_line) + '/train/masks_lines/' + str(patch_counter_train) + '.png')),
                            p_masks_line[j])
                patch_counter_train += 1

        if train_images_img.shape[0]%256 is not 0:
            patches_img = patch_bottomborder(train_images_img, PATCH_SIZE)
            patches_zones = patch_bottomborder(train_zones_img, PATCH_SIZE)
            patches_lines = patch_bottomborder(train_front_img, PATCH_SIZE)
            for number in range(patches_img.shape[0]):
                cv2.imwrite(
                    str(Path(
                        'data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/train/images/' + str(patch_counter_train) + '.png')),
                    patches_img[number,:])
                cv2.imwrite(str(
                    Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/train/masks_zones/' + str(
                        patch_counter_train) + '.png')),
                    patches_zones[number,:])

                cv2.imwrite(str(
                    Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/train/masks_lines/' + str(
                        patch_counter_train) + '.png')),
                    patches_lines[number,:])
                patch_counter_train += 1

        if train_images_img.shape[1]%256 is not 0:
            patches_img = patch_rightborder(train_images_img, PATCH_SIZE)
            patches_zones = patch_rightborder(train_zones_img, PATCH_SIZE)
            patches_lines = patch_rightborder(train_front_img, PATCH_SIZE)
            for number in range(patches_img.shape[0]):
                cv2.imwrite(
                    str(Path(
                        'data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/train/images/' + str(patch_counter_train) + '.png')),
                    patches_img[number,:])
                cv2.imwrite(str(
                    Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/train/masks_zones/' + str(
                        patch_counter_train) + '.png')),
                    patches_zones[number,:])
                cv2.imwrite(str(
                    Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/train/masks_lines/' + str(
                        patch_counter_train) + '.png')),
                    patches_lines[number,:])
                patch_counter_train += 1

# validation path
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/val/images'))): os.makedirs(
    str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/val/images')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/val/masks_zones'))): os.makedirs(
    str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/val/masks_zones')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/val/masks_lines'))): os.makedirs(
    str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/val/masks_lines')))

STRIDE_val = (PATCH_SIZE, PATCH_SIZE)
patch_counter_val = 0

print("Creating Val images")
for i in range(len(val_images_id)):
    val_images_img = imageio.imread(val_images_id.iloc[i])
    val_images_img = apply_filter(pre_name, val_images_img)
    val_zones_img = imageio.imread(val_zones_id.iloc[i])
    val_front_img = imageio.imread(val_front_id.iloc[i])

    val_images_img = cv2.copyMakeBorder(val_images_img, 0, (PATCH_SIZE - val_images_img.shape[0]) % PATCH_SIZE, 0,
                                          (PATCH_SIZE - val_images_img.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
    val_zones_img = cv2.copyMakeBorder(val_zones_img, 0, (PATCH_SIZE - val_zones_img.shape[0]) % PATCH_SIZE, 0,
                                         (PATCH_SIZE - val_zones_img.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
    val_front_img = cv2.copyMakeBorder(val_front_img, 0, (PATCH_SIZE - val_front_img.shape[0]) % PATCH_SIZE, 0,
                                         (PATCH_SIZE - val_front_img.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)

    masks_zone_tmp = val_zones_img
    masks_zone_tmp[masks_zone_tmp == 127] = 0
    masks_zone_tmp[masks_zone_tmp == 254] = 255

    # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement

    p_masks_zone, i_masks_zone = Amir_utils.extract_grayscale_patches(masks_zone_tmp, (PATCH_SIZE, PATCH_SIZE),
                                                                      stride=STRIDE_val)
    p_img, i_img = Amir_utils.extract_grayscale_patches(val_images_img, (PATCH_SIZE, PATCH_SIZE),
                                                        stride=STRIDE_val)
    p_masks_line, i_masks_line = Amir_utils.extract_grayscale_patches(val_front_img,
                                                                      (PATCH_SIZE, PATCH_SIZE), stride=STRIDE_val)

    for j in range(p_masks_zone.shape[0]):
        # if np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) > 0.05 and np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) < 0.95: # only those patches that has both background and foreground
        if np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) > 0 and np.count_nonzero(p_masks_zone[j]) / (
                PATCH_SIZE * PATCH_SIZE) < 1:
            cv2.imwrite(
                str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/val/images/' + str(patch_counter_val) + '.png')),
                p_img[j])
            cv2.imwrite(
                str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/val/masks_zones/' + str(patch_counter_val) + '.png')),
                p_masks_zone[j])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            p_masks_line[j] = cv2.dilate(p_masks_line[j], kernel, iterations=thick_line)
            cv2.imwrite(
                str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/val/masks_lines/' + str(patch_counter_val) + '.png')),
                p_masks_line[j])
            patch_counter_val += 1
            # store the name of the file that the patch is from in a list as well

#####
# test path
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/test/images'))): os.makedirs(
    str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/test/images')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/test/masks_zones'))): os.makedirs(
    str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/test/masks_zones')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/test/masks_lines'))): os.makedirs(
    str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/test/masks_lines')))

STRIDE_test = (PATCH_SIZE, PATCH_SIZE)
patch_counter_test = 0
print("Creating Test images")
for i in range(len(test_images_id)):
    test_images_img = imageio.imread(test_images_id.iloc[i])
    test_images_img = apply_filter(pre_name, test_images_img)
    test_zones_img = imageio.imread(test_zones_id.iloc[i])
    test_front_img = imageio.imread(test_front_id.iloc[i])

    masks_zone_tmp = test_zones_img
    masks_zone_tmp[masks_zone_tmp == 127] = 0
    masks_zone_tmp[masks_zone_tmp == 254] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    test_front_img = cv2.dilate(test_front_img, kernel, iterations=thick_line)

    cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/test/images/' + test_images_id.iloc[i].split('/')[-1])),
                test_images_img)
    cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/test/masks_zones/' + test_zones_id.iloc[i].split('/')[-1])),
                masks_zone_tmp)
    cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + pre_name +'_linesw'+str(size)+'i'+str(thick_line)+ '/test/masks_lines/' + test_front_id.iloc[i].split('/')[-1])),
                test_front_img)


#train_augmentation
def apply_aug(image, name):
    if name == 'horizontal':
        image = cv2.flip(image, 0)
    elif name == 'rotate90':
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif name == 'rotate180':
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif name == 'rotate270':
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif name == 'hor_rotate90':
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif name == 'hor_rotate180':
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif name == 'hor_rotate270':
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        print('some error happened on transformation')
    return image

pathimage = str(Path('data_'+str(PATCH_SIZE)+pre_name+'_linesw'+str(size)+'i'+str(thick_line)+'/train/images'))
pathzones = str(Path('data_'+str(PATCH_SIZE)+pre_name+'_linesw'+str(size)+'i'+str(thick_line)+'/train/masks_zones'))
pathlines = str(Path('data_'+str(PATCH_SIZE)+pre_name+'_linesw'+str(size)+'i'+str(thick_line)+'/train/masks_lines'))

all_images = os.listdir(pathimage)

augments = ['horizontal','rotate90','rotate180', 'rotate270', 'hor_rotate90', 'hor_rotate180', 'hor_rotate270']

if augment:
    for augnames in augments[0:num_augments]:
        print("applying offline augmentations = ", augnames)
        for i, image_names in enumerate(all_images):
            image = imageio.imread(str(Path(pathimage , image_names)))
            zones = imageio.imread(str(Path(pathzones, image_names)))
            lines = imageio.imread(str(Path(pathlines, image_names)))

            img_aug = apply_aug(image, augnames)
            zones_aug = apply_aug(zones, augnames)
            lines_aug = apply_aug(lines, augnames)

            num_name = str(i)+augnames
            cv2.imwrite(str(Path(pathimage, num_name, '.png')), img_aug)
            cv2.imwrite(str(Path(pathzones,num_name,'.png')), zones_aug)
            cv2.imwrite(str(Path(pathlines, num_name, '.png')), lines_aug)
else:
    print('no augmentations')

print(len(os.listdir(pathimage)))







