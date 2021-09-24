from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import gaussian_filter
import os
import argparse

import numpy as np
parser = argparse.ArgumentParser(description='create distance maps for visualization/dmap algorithm')
parser.add_argument('--work_dir', default='distance_maps_lines', type=str, help='source directory')
parser.add_argument('--image_name', default='86horizontal.png', type=str, help='image to be tested on')
parser.add_argument('--w', default=15, type=int, help='parameter w')
parser.add_argument('--R', default=8, type=int, help='parameter R')
parser.add_argument('--k', default=0.1, type=float, help='parameter k')

args = parser.parse_args()

work_dir = Path(args.work_dir)
image_name = args.image_name

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

gt_path = Path(str(work_dir), image_name)
y = imageio.imread(str(gt_path))
y = y/255

plt.imshow(y, cmap='gray')
plt.axis('off')
plt.savefig(r'distance_maps_lines/GT.png', bbox_inches='tight', format='png', dpi=800)
#plt.show()

w= args.w
R= args.R
k= args.k

if y.max() == 0:
    soft_gt = 1 - y
    print(soft_gt.max(), soft_gt.min())
else:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, w))
    fat_gt = cv2.dilate(y, kernel, iterations=1)
    soft_gt = edt(fat_gt)/R
    soft_gt = 1 / (1 + np.exp(-soft_gt))
    soft_gt = soft_gt - soft_gt.min()
    soft_gt = soft_gt / soft_gt.max()
    print(soft_gt.max(), soft_gt.min())
    #fat_gt = fat_gt - fat_gt.min()
    #fat_gt = fat_gt / fat_gt.max()
    inv_gt = 1 - fat_gt
    soft_gt = soft_gt + inv_gt*k
    print(soft_gt.max(), soft_gt.min())

#soft_gt = 5*soft_gt
#soft_gt = soft_gt/soft_gt.max()
#soft_gt[soft_gt==0]=1
#sig_gt = sigmoid(soft_gt)
plt.imshow(soft_gt, cmap='gray')
plt.axis('off')
plt.savefig(str(Path('distance_maps_lines', 'wRk_'+ str(w)+'_'+str(R)+'_'+str(k)+'.png')), bbox_inches='tight', format='png', dpi=800)
