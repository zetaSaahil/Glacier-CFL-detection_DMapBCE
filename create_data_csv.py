from pathlib import Path
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='create csv of train/val/test')
parser.add_argument('--source', default=r'../JHN', type=str, help='fource folder of images data')
parser.add_argument('--random_state', default=1, type=int, help='random state for train, val, test split')
parser.add_argument('--test_size', default=25, type=int, help='val/test size for split')
args = parser.parse_args()
source = args.source

random_state = args.random_state
test_size = args.test_size

all_images = []
all_zones = []
all_lines = []

for filename in Path(source).rglob('*.png'):
    if '_NA.png' in filename.as_posix():
        continue
    elif '_zones.png' in filename.as_posix():
        all_zones.append(filename.as_posix())
    elif '_front.png' in filename.as_posix():
        all_lines.append(filename.as_posix())
    else:
        all_images.append(filename.as_posix())

train_img, test_img = train_test_split(all_images, test_size=test_size, random_state=random_state) # test split for images
train_img, val_img = train_test_split(train_img, test_size=test_size, random_state=random_state) # train val split for images

train_zones, test_zones = train_test_split(all_zones, test_size=test_size, random_state=random_state)  # test split for zones
train_zones, val_zones = train_test_split(train_zones, test_size=test_size, random_state=random_state) # train val split for zones

train_lines, test_lines = train_test_split(all_lines, test_size=test_size, random_state=random_state) # test split for lines
train_lines, val_lines = train_test_split(train_lines, test_size=test_size, random_state=random_state) # train val split for lines



#train_d = {'images': list(map(str, train_img)), 'masks':list(map(str, train_zones)), 'lines':list(map(str, train_lines))}
train_d = {'images': train_img, 'masks':train_zones, 'lines':train_lines}
train_df = pd.DataFrame(data=train_d)

#val_d = {'images': list(map(str, val_img)), 'masks':list(map(str, val_zones)), 'lines':list(map(str, val_lines))}
val_d = {'images': val_img, 'masks':val_zones, 'lines':val_lines}
val_df = pd.DataFrame(data=val_d)

#test_d = {'images': list(map(str, test_img)), 'masks':list(map(str, test_zones)), 'lines':list(map(str, test_lines))}
test_d = {'images': test_img, 'masks':test_zones, 'lines':test_lines}
test_df = pd.DataFrame(data=test_d)

train_df.to_csv('train_images.csv', header=True)
val_df.to_csv('validation_images.csv', header=True)
test_df.to_csv('test_images.csv', header=True)