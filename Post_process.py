import argparse
from pathlib import Path
import imageio
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2
import argparse

parser = argparse.ArgumentParser(description='post_process zones')
parser.add_argument('--source_dir', default='data_256median_aug2_thin_lines/test', type=str, help='primary source directory based on filter/augmentation, this path must contain the ground truth for lines for metrics calculation')
parser.add_argument('--pred_dir', default='masks_predicted_binary_crossentropy_monitor_mcc', type=str, help= 'secondary directory i.e. predicted zones directory')
args = parser.parse_args()

# remove false alarms from predicted zones
def remove_false_alarms(norm_image):
    areas = np.array([])
    refined_norm_image = np.zeros_like(norm_image)
    labels = measure.label(norm_image)
    regions = measure.regionprops(labels)
    for props in regions:
        areas = np.append(areas, props.area)
        if props.area == areas.max():
            coordinates = props.coords

    for coords in coordinates:
        refined_norm_image[coords[0], coords[1]] = 1

    return refined_norm_image

def metrics_binary(pred, gtruth):
    k = 255
    tp = float(np.sum(pred[gtruth == k] == k))
    tn = float(np.sum(pred[gtruth == 0] == 0))
    fp = float(np.sum(pred[gtruth == 0] == k))
    fn = float(np.sum(pred[gtruth == k] == 0))


    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn)

    intersection = np.logical_and(gtruth, pred)
    union = np.logical_or(gtruth, pred)
    iou_score = np.sum(intersection) / np.sum(union)

    numerator = (tp * tn - fp * fn)
    denominator = np.sqrt(np.abs((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc_val = numerator / (denominator + 1e-9)

    return pixel_accuracy, iou_score, dice, mcc_val

source = args.source_dir

pred_source = Path(source,args.pred_dir)

all_preds = []
all_masks = []
for filenames in pred_source.rglob('*.png'):
    all_preds.append(filenames.as_posix())

for filenames in Path(source, 'masks_lines').rglob('*.png'):
    all_masks.append(filenames.as_posix())

if not Path.exists(Path(str(pred_source), 'Post_Processed')): Path.mkdir(Path(str(pred_source), 'Post_Processed'))
if not Path.exists(Path(str(pred_source), 'Post_Processed/Edges')): Path.mkdir(Path(str(pred_source), 'Post_Processed/Edges'))

pixaccuracy_all = []
iou_all = []
dice_all_final = []
mcc_all = []

for i in range(len(all_masks)):
    #print(all_preds[i])
    #print(all_masks[i])
    pred = imageio.imread(all_preds[i])
    mask = imageio.imread(all_masks[i])
    pred = pred/pred.max()
    print((all_preds[i].split('/')[-1]).split('.')[0])

    refined_norm_image = remove_false_alarms(pred)

    inv_img = 1 - refined_norm_image
    final_image = 1 - remove_false_alarms(inv_img)
    final_image = (final_image * 255).astype(np.uint8)

    pix_accuracy, iou, dice, mcc_val = metrics_binary(final_image, mask)
    pixaccuracy_all.append(pix_accuracy)
    iou_all.append(iou)
    dice_all_final.append(dice)
    mcc_all.append(mcc_val)

    cv2.imwrite(str(Path(str(pred_source), r'Post_Processed', str((all_preds[i].split('/')[-1]).split('.')[0]) + '.png')), final_image)

    erode_kernel = np.ones((4, 4))
    eroded = cv2.erode(final_image, erode_kernel)
    # edge_image = cv2.Canny(final_image, 100, 100)
    edge_image = final_image - eroded
    cv2.imwrite(str(Path(str(pred_source), r'Post_Processed/Edges', str((all_preds[i].split('/')[-1]).split('.')[0]) + '.png')), edge_image) # edge detection on postprocessed zones

pixaccuracy_all_avg = np.mean(pixaccuracy_all)
iou_all_avg = np.mean(iou_all)
dice_all_avg = np.mean(dice_all_final)
mcc_avg = np.mean(mcc_all)


#with open(r'Post_processed/ReportOnModel.txt', 'w') as f:
#    f.write('accuracy = ' + str(pixaccuracy_all_avg) + '\n')
#    f.write('iou = ' + str(iou_all_avg) + '\n')
#    f.write('dice = ' + str(dice_all_avg) + '\n')
#    f.write('mcc = ' + str(mcc_avg) + '\n')
#    f.close()