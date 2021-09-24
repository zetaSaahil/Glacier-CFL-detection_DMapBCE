from pathlib import Path
import imageio
import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt as edt
import argparse

parser = argparse.ArgumentParser(description='calculate quantitative results for a list of tolerance levels')
parser.add_argument('--source_dir', default='data_256median_aug2_thin_lines/test', type=str, help='primary source directory based on filter/augmentation')
parser.add_argument('--pred_dir', default='masks_lines_predicted_d_map_bce_monitor_val_mcc_w3R1k0.1', type=str, help= 'secondary directory i.e. predicted lines directory')
args = parser.parse_args()

source = args.source_dir

pred_source = Path(source, args.pred_dir)

tolerance_list = [2,3,5,6, 8, 10, 12, 13] #list of tolerance levels (These are filter sizes for dilation)




all_preds = []
all_masks = []
for filenames in pred_source.rglob('*.png'):
    all_preds.append(filenames.as_posix())

for filenames in Path(source, 'masks_lines').rglob('*.png'):
    all_masks.append(filenames.as_posix())

def metrics_binary(pred, gtruth):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fat_gt = cv2.dilate(gtruth, kernel, iterations=1)
    soft_gt = edt(fat_gt) / 1
    soft_gt = 1 / (1 + np.exp(-soft_gt))
    soft_gt = soft_gt - soft_gt.min()
    soft_gt = soft_gt / soft_gt.max()
    inv_gt = 1 - fat_gt
    soft_gt = soft_gt + inv_gt

    y_pred = soft_gt*pred
    tp = np.sum(gtruth * y_pred)
    tn = np.sum((1 - gtruth) * (1 - y_pred))
    fp = np.sum((1 - gtruth) * y_pred)
    fn = np.sum(gtruth * (1 - y_pred))


    dice = 2 * tp / (2 * tp + fp + fn + 1e-16)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-16)

    intersection = np.logical_and(gtruth, pred)
    union = np.logical_or(gtruth, pred)
    iou_score = np.sum(intersection) / np.sum(union)

    numerator = (tp * tn - fp * fn)
    denominator = np.sqrt(np.abs((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc_val = numerator / (denominator + np.finfo(np.float32).eps)

    return pixel_accuracy, iou_score, dice, mcc_val

all_acc = []
all_iou = []
all_dice = []
all_mcc = []


with open(str(Path(str(pred_source), 'RelaxedMetrics.txt')), 'a') as f:
    f.write('at tolerance of __ pixels = overlap_acc, dilated_dice, dilated_iou, dilated_mcc  ' + '\n')
    f.write('\n')

for relax in tolerance_list: # list of tolerance i.e. filter size for thickening lines
    print('relaxing upto ' + str(relax + 4) + ' pixels')
    all_overlap_p = []
    all_dil_dice = []
    all_dil_iou = []
    all_dil_mcc = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (relax, relax))
    for i in range(len(all_masks)):
        #print(all_preds[i])
        #print(all_masks[i])
        pred = imageio.imread(all_preds[i])
        mask = imageio.imread(all_masks[i])
        pred = pred/pred.max()
        mask = mask/mask.max()


        fat_mask = cv2.dilate(mask, kernel)
        dilated_pred = cv2.dilate(pred, kernel)

        tp = np.sum(fat_mask * dilated_pred)
        tn = np.sum((1 - fat_mask) * (1 - dilated_pred))
        fp = np.sum((1 - fat_mask) * dilated_pred)
        fn = np.sum(fat_mask * (1 - dilated_pred))

        pred_all_white_pixels = np.sum(pred)
        mask_all_white_pixels = np.sum(mask)

        #overlap = pred + fat_mask
        #overlap = overlap/overlap.max()
        overlap_all_white_pixels = np.sum(pred * fat_mask)
        overlap_acc_p = overlap_all_white_pixels/pred_all_white_pixels

        dilated_dice = 2*np.sum(dilated_pred*fat_mask)/(np.sum(dilated_pred) + np.sum(fat_mask))
        #overlap_dice = 2*np.sum(dilated_pred*mask)/(pred_all_white_pixels+mask_all_white_pixels)
        intersection = np.logical_and(fat_mask, dilated_pred)
        union = np.logical_or(fat_mask, dilated_pred)
        dilated_iou = np.sum(intersection) / np.sum(union)

        numerator = (tp * tn - fp * fn)
        denominator = np.sqrt(np.abs((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        dilated_mcc = numerator / (denominator + np.finfo(np.float32).eps)
        #print(overlap_acc)
        all_overlap_p.append(overlap_acc_p)
        all_dil_dice.append(dilated_dice)
        all_dil_iou.append(dilated_iou)
        all_dil_mcc.append(dilated_mcc)


    mean_ov_p = np.mean(all_overlap_p)
    mean_dil_dice = np.mean(all_dil_dice)
    mean_dil_iou = np.mean(all_dil_iou)
    mean_dil_mcc = np.mean(all_dil_mcc)

    print(mean_ov_p, mean_dil_iou, mean_dil_dice, mean_dil_mcc)

    with open(str(Path(str(pred_source), 'RelaxedMetrics.txt')), 'a') as f:
        f.write('at tolerance of ' + str(relax + 4) + ' pixels = ' + str(mean_ov_p) + ', ' + str(mean_dil_dice) + ', ' + str(mean_dil_iou) + ', ' + str(mean_dil_mcc) + '\n')
