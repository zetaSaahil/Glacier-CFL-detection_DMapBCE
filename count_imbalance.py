from pathlib import Path
import imageio
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='count imbalance')
parser.add_argument('--source', default=r'\train\masks_lines', type=str, help='source folder for counting imbalance')
args = parser.parse_args()
source = args.source



count_ones = 0
count_zeros = 0
for files in Path(source).rglob('*.png'):
    print(files)
    img = imageio.imread(files)
    img = img/255
    print(np.max(img), np.min(img))
    ones = (img == 1).sum()
    zeros = (img == 0).sum()
    count_ones = count_ones + ones
    count_zeros = count_zeros + zeros

print('zeros = ' + str(count_zeros))
print('ones = ' +  str(count_ones))
print('ratio = ' + str(np.round(count_zeros/count_ones)))