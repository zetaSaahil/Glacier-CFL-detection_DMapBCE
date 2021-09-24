# -*- coding: utf-8 -*-

from model import unet, unet_Enze19, unet_Enze19_2
from data import trainGenerator, testGenerator, saveResult, saveResult_Amir
import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.optimizers import *
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib.use('ps')
import cv2
import os
import time
from scipy.spatial import distance
import argparse
from Loss import *
from Utils import custom_earlystop, CyclicLR, remove_images, apply_filter
import sys
from keras.models import load_model


# %% Hyper-parameter tuning
parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

# parser.add_argument('--Test_Size', default=0.2, type=float, help='Test set ratio for splitting from the whole dataset (values between 0 and 1)')
# parser.add_argument('--Validation_Size', default=0.1, type=float, help='Validation set ratio for splitting from the training set (values between 0 and 1)')

# parser.add_argument('--Classifier', default='unet', type=str, help='Classifier to use (unet/unet_Enze19)')
parser.add_argument('--Epochs', default=400, type=int, help='number of training epochs (integer value > 0) per run')
parser.add_argument('--Batch_Size', default=15, type=int, help='batch size (integer value)')
parser.add_argument('--Patch_Size', default=256, type=int, help='Patch size (integer value)')
parser.add_argument('--patience', default=35, type=int, help='patience for earlystop')

# parser.add_argument('--EARLY_STOPPING', default=1, type=int, help='If 1, classifier is using early stopping based on validation loss with patience 20 (0/1)')
parser.add_argument('--LOSS', default='binary_crossentropy', type=str,
                    help='binary_crossentropy, focal_loss, dice_loss, mcc_loss1, mcc_loss2, mcc_loss3,weighted_bce,d_map_mcc,d_map_bce,d_map_weighted_bce')

parser.add_argument('--OUTPATH', default='output/results/21.09/', type=str, help='Output path for results')
parser.add_argument('--filtname', default='median', type=str, help='preprocessing filter name')
parser.add_argument('--max_runs', default=1, type=int, help='total max num of runs to be performed sequentially')
parser.add_argument('--mcc_eta', default=10, type=float, help='additional params for mcc loss')
parser.add_argument('--monitor', default='val_loss', type=str, help='monitor metric for early stop')
parser.add_argument('--input_path', default='data_256median_linesw3i4', type=str, help='secondary path argument')
parser.add_argument('--relax', default=4, type=int, help='relaxation of d map losses')
parser.add_argument('--class_weights', default=418, type=float, help='class weights for weighted bce')
parser.add_argument('--dilation', default=3, type=int, help='amount of dilation of the lines')

# parser.add_argument('--Random_Seed', default=1, type=int, help='random seed number value (any integer value)')
args = parser.parse_args()

# %%
START = time.time()

PATCH_SIZE = args.Patch_Size
batch_size = args.Batch_Size
lossname = args.LOSS
patience = args.patience

pre_name = args.filtname
max_runs = args.max_runs
mcc_eta = args.mcc_eta
monitor = args.monitor
input_path  = args.input_path

relax = args.relax

thick_line = args.dilation
#print(thick_line)

if lossname == 'focal_loss':
    loss = focal_loss
elif lossname == 'dice_loss':
    loss = dice_loss
elif lossname == 'mcc_loss1':
    loss = mcc_loss1(eta=mcc_eta)
elif lossname == 'mcc_loss2':
    loss = mcc_loss2(eta=mcc_eta)
elif lossname == 'mcc_loss3':
    loss = mcc_loss3(eta=mcc_eta)
elif lossname == 'binary_crossentropy':
    loss = 'binary_crossentropy'
elif lossname == 'weighted_bce':
    loss = weighted_bce(w=args.class_weights)
elif lossname == 'd_map_mcc':
    loss = d_map_mcc(relax=relax)
elif lossname == 'd_map_bce':
    loss = d_map_bce(relax=relax)
elif lossname == 'd_map_weighted_bce':
    loss = d_map_weighted_bce(w=args.class_weights, relax=relax)
else:
    sys.exit('wrong loss name')



num_samples = len([file for file in Path(input_path + '/train/images/').rglob(
    '*.png')])  # number of training samples
num_val_samples = len([file for file in Path(input_path + '/val/images/').rglob(
    '*.png')])  # number of validation samples

if lossname == 'd_map_mcc' or lossname == 'd_map_bce' or lossname == 'd_map_weighted_bce':
    Out_Path = Path(input_path + '/test/masks_lines_predicted_' + lossname + '_monitor_' + monitor + '_R' + str(relax))  # adding the time in the folder name helps to keep the results for multiple back to back exequations of the code
else:
    Out_Path = Path(input_path + '/test/masks_lines_predicted_' + lossname + '_monitor_' + monitor)  # adding the time in the folder name helps to keep the results for multiple back to back exequations of the code

if not os.path.exists(Path('data/train/aug')): os.makedirs(Path('data/train/aug'))
if not os.path.exists(Out_Path): os.makedirs(Out_Path)



# data_gen_args = dict(rotation_range=0.2,
#                    width_shift_range=0.05,
#                    height_shift_range=0.05,
#                    shear_range=0.05,
#                    zoom_range=0.05,
#                    horizontal_flip=True,
#                    fill_mode='nearest')
#data_gen_args = dict(horizontal_flip=True, vertical_flip=True)

train_Generator = trainGenerator(batch_size=batch_size,
                                 train_path=str(Path(input_path +  '/train')),
                                 image_folder='images',
                                 mask_folder='masks_lines',
                                 aug_dict=None,
                                 save_to_dir=None)

val_Generator = trainGenerator(batch_size=batch_size,
                               train_path=str(Path(input_path + '/val')),
                               image_folder='images',
                               mask_folder='masks_lines',
                               aug_dict=None,
                               save_to_dir=None)

import keras.backend as K


def mcc(y_true, y_pred):
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def bce(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred)


model = unet_Enze19_2()
init_epoch = 0
no_of_runs = 1
learning_rate = 1e-2
val_acc = None
val_mcc = None
val_loss = None
val_bce = None
if os.path.exists(Path(str(Out_Path) + '//' + 'history.csv')):
    old_hist = pd.read_csv(str(Out_Path) + '//' + 'history.csv')
    init_epoch = len(old_hist.index)
    no_of_runs = int(np.floor(init_epoch / args.Epochs) + 1)
    if init_epoch%args.Epochs is not 0 or no_of_runs>max_runs:
        sys.exit('early stop occured, no more training further / reached max epoch ... change name of previous run for new training')
    val_acc = old_hist['val_accuracy'].to_numpy()
    val_mcc = old_hist['val_mcc'].to_numpy()
    val_loss = old_hist['val_loss'].to_numpy()
    val_bce = old_hist['val_bce'].to_numpy()


print('no of runs = ' + str(no_of_runs))
print('initial epoch = ' + str(init_epoch))
print('learning rate = ' + str(learning_rate))

model.compile(optimizer=Adam(lr=learning_rate), loss=[loss], metrics=['accuracy', mcc, bce])

if no_of_runs > 1:
    model = keras.models.load_model(str(Path(str(Out_Path) + '/unet_zone.h5')),
                                    custom_objects={'mcc': mcc, 'bce':bce,'focal_loss': focal_loss, 'dice_loss': dice_loss, 'mcc_loss1': mcc_loss1, 'mcc_loss2':mcc_loss2, 'mcc_loss3':mcc_loss3, 'mcc_loss_fixed':loss, 'weighted_bce':weighted_bce, 'bce_fixed':loss, 'd_map_mcc':d_map_mcc, 'd_map_mcc_fixed':loss,'d_map_bce':d_map_bce, 'd_map_bce_fixed':loss, 'd_map_weighted_bce':d_map_weighted_bce,'d_map_weighted_bce_fixed':loss,'calc_dist_map':calc_dist_map, 'calc_dist_map_batch':calc_dist_map_batch})
    #model.compile(optimizer=Adam(lr=learning_rate), loss=[loss], metrics=['accuracy', mcc])

if monitor == 'val_loss':
    log_prev = val_loss
    mode = 'min'
elif monitor == 'val_mcc':
    log_prev = val_mcc
    mode = 'max'
elif monitor == 'val_bce':
    log_prev = val_bce
    mode = 'min'
else:
    log_prev = val_acc
    mode = 'max'

model_checkpoint = ModelCheckpoint(str(Path(str(Out_Path), 'unet_zone.h5')), monitor=monitor, mode=mode, verbose=0,
                                   save_best_only=True)
earlystop = EarlyStopping(monitor=monitor, min_delta=1e-7, patience=patience, verbose=0, mode=mode, baseline=None,
                          restore_best_weights=True)


custom_es = custom_earlystop(monitor=monitor, patience= patience, previous_historylog=log_prev, prev_bestweights=model.get_weights())

steps_per_epoch = np.ceil(num_samples / batch_size)
validation_steps = np.ceil(num_val_samples / batch_size)

cyclic_lr = CyclicLR(base_lr=1e-8,
            max_lr=1e-4,
            step_size=5*steps_per_epoch,)



History = model.fit_generator(train_Generator,
                              steps_per_epoch=steps_per_epoch, initial_epoch=init_epoch,
                              epochs=args.Epochs + init_epoch,
                              validation_data=val_Generator,
                              validation_steps=validation_steps,
                              verbose=1,
                              callbacks=[model_checkpoint, custom_es, cyclic_lr])
# model.fit_generator(train_Generator, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint], class_weight=[0.0000001,0.9999999])
print(History.history.keys())
hist_df = pd.DataFrame(History.history)

hist_csv_file = str(Out_Path) + '//' + 'history.csv'
if os.path.exists(Path(str(Out_Path) + '//' + 'history.csv')):
    new_hist = old_hist.append(hist_df, sort=False, ignore_index=True)
    os.remove(hist_csv_file)
    with open(hist_csv_file, mode='w') as f:
        new_hist.to_csv(f, index=False)

else:
    new_hist = hist_df
    with open(hist_csv_file, mode='w') as f:
        new_hist.to_csv(f, index=False)

##########
##########
# save loss plot

if no_of_runs >= max_runs or len(new_hist.index)%args.Epochs is not 0:

    total_epochs = np.arange(1, len(new_hist.index) + 1)
    t_loss = new_hist["loss"].to_numpy()
    t_acc = new_hist["accuracy"].to_numpy()
    t_mcc = new_hist["mcc"].to_numpy()
    v_loss = new_hist["val_loss"].to_numpy()
    v_acc = new_hist["val_accuracy"].to_numpy()
    v_mcc = new_hist["val_mcc"].to_numpy()
    v_bce = new_hist['val_bce'].to_numpy()

    plt.figure(1)
    plt.rcParams.update({'font.size': 18})
    plt.plot(total_epochs, t_loss, 'X-', label='training loss', linewidth=2.0)
    plt.plot(total_epochs, v_loss, 'o-', label='validation loss', linewidth=2.0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--')
    plt.savefig(str(Path(str(Out_Path), 'loss_plot.png')), bbox_inches='tight', format='png', dpi=200)

    plt.figure(2)
    plt.rcParams.update({'font.size': 18})
    plt.plot(total_epochs, t_acc, 'X-', label='training accuracy', linewidth=2.0)
    plt.plot(total_epochs, v_acc, 'o-', label='validation accuracy', linewidth=2.0)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--')
    plt.savefig(str(Path(str(Out_Path), 'acc_plot.png')), bbox_inches='tight', format='png', dpi=200)

    plt.figure(3)
    plt.rcParams.update({'font.size': 18})
    plt.plot(total_epochs, t_mcc, 'X-', label='training mcc', linewidth=2.0)
    plt.plot(total_epochs, v_mcc, 'o-', label='validation mcc', linewidth=2.0)
    plt.xlabel('epoch')
    plt.ylabel('mcc')
    plt.legend(loc='lower right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--')
    plt.savefig(str(Path(str(Out_Path), 'mcc_plot.png')), bbox_inches='tight', format='png', dpi=200)

    # # save model
    # model.save(str(Path(OUTPUT_PATH + 'MyModel' + SaveName + '.h5').absolute()))
    ##########
    ##########

    # %%
    # testGene = testGenerator('data_256/test/images', num_image=5) #########
    # results = model.predict_generator(testGene, 5, verbose=0) #########
    # saveResult_Amir('data_256/test/masks_predicted', results) #########

    #####################
    #####################
    import skimage.io as io
    import Amir_utils
    test_path = str(Path(input_path +  '/test/'))

    DICE_all = []
    EUCL_all = []
    pixacc_all = []
    iou_all = []
    mcc_all = []
    soft_acc_all = []
    soft_iou_all = []
    soft_mcc_all = []
    soft_dice_all = []

    dice_all_threshold = []
    test_file_names = []
    Perf = {}


    def metrics_binary(pred, gtruth):
        k = 255
        tp = float(np.sum(pred[gtruth == k] == k))
        tn = float(np.sum(pred[gtruth == 0] == 0))
        fp = float(np.sum(pred[gtruth == 0] == k))
        fn = float(np.sum(pred[gtruth == k] == 0))


        dice = 2 * tp / (2 * tp + fp + fn + 1e-16)
        pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-16)

        intersection = np.logical_and(gtruth, pred)
        union = np.logical_or(gtruth, pred)
        iou_score = np.sum(intersection) / np.sum(union)

        numerator = (tp * tn - fp * fn)
        denominator = np.sqrt(np.abs((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc_val = numerator / (denominator + np.finfo(np.float32).eps)

        return pixel_accuracy, iou_score, dice, mcc_val

    def soft_metrics(pred, gtruth):
        kernel = np.ones((5, 5))
        pred = cv2.dilate(pred, kernel, iterations=2)
        gtruth = cv2.dilate(gtruth, kernel, iterations=2)

        k = 255
        tp = float(np.sum(pred[gtruth == k] == k))
        tn = float(np.sum(pred[gtruth == 0] == 0))
        fp = float(np.sum(pred[gtruth == 0] == k))
        fn = float(np.sum(pred[gtruth == k] == 0))

        dice = 2 * tp / (2 * tp + fp + fn + 1e-16)
        pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-16)

        intersection = np.logical_and(gtruth, pred)
        union = np.logical_or(gtruth, pred)
        iou_score = np.sum(intersection) / np.sum(union)

        numerator = (tp * tn - fp * fn)
        denominator = np.sqrt(np.abs((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc_val = numerator / (denominator + np.finfo(np.float32).eps)

        return pixel_accuracy, iou_score, dice, mcc_val


    import imageio
    valcsv_path = 'validation_images.csv'
    df_val = pd.read_csv(valcsv_path)
    #df_val = remove_images(df_val, 5)
    #df_val = remove_images(df_val, 6)
    val_images_id = df_val.iloc[0:, 1]
    val_lines_id = df_val.iloc[0:, 3]

    thresholds = np.arange(0, 1, 0.05)
    thresholds = np.append(thresholds, 1)
    for thres in thresholds:
        print(f'threshold check at {thres:.2f}')
        dice_all_images = []
        for i in range(len(val_images_id)):
            val_images_img = imageio.imread(val_images_id.iloc[i], as_gray=True)
            val_images_img = apply_filter(pre_name, val_images_img)
            val_lines_img = imageio.imread(val_lines_id.iloc[i], as_gray=True)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            val_lines_img = cv2.dilate(val_lines_img, kernel, iterations=thick_line)

            img = val_images_img
            img = img / 255
            img_pad = cv2.copyMakeBorder(img, 0, (PATCH_SIZE - img.shape[0]) % PATCH_SIZE, 0,
                                         (PATCH_SIZE - img.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
            p_img, i_img = Amir_utils.extract_grayscale_patches(img_pad, (PATCH_SIZE, PATCH_SIZE),
                                                                stride=(PATCH_SIZE, PATCH_SIZE))
            p_img = np.reshape(p_img, p_img.shape + (1,))
            p_img_predicted = model.predict(p_img)

            p_img_predicted = np.reshape(p_img_predicted, p_img_predicted.shape[:-1])
            img_mask_predicted_recons = Amir_utils.reconstruct_from_grayscale_patches(p_img_predicted, i_img)[0]

            # unpad and normalize
            img_mask_predicted_recons_unpad = img_mask_predicted_recons[0:img.shape[0], 0:img.shape[1]]
            img_mask_predicted_recons_unpad_norm = cv2.normalize(src=img_mask_predicted_recons_unpad, dst=None, alpha=0,
                                                                 beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # quantization to make the binary masks

            img_mask_predicted_recons_unpad_norm[img_mask_predicted_recons_unpad_norm < thres * 255] = 0
            img_mask_predicted_recons_unpad_norm[img_mask_predicted_recons_unpad_norm >= thres * 255] = 255

            # DICE
            gt = val_lines_img
            #gt[gt == 127] = 0
            #gt[gt == 254] = 255
            pix_acc, iou, dice, mcc_val = metrics_binary(img_mask_predicted_recons_unpad_norm, gt)
            dice_all_images.append(dice)

        dice_all_threshold.append(np.mean(dice_all_images))

    dice_all_threshold = np.array(dice_all_threshold)
    best_dice = dice_all_threshold.max()
    print(dice_all_threshold)
    best_threshold = thresholds[dice_all_threshold.argmax()]

    plt.figure(4)
    plt.plot(thresholds, dice_all_threshold)
    plt.vlines(x=best_threshold, ymin=dice_all_threshold.min(), ymax=dice_all_threshold.max())
    plt.text(best_threshold + 0.003, best_dice - 0.001, f'DICE = {best_dice:.3f}', fontsize=11)
    plt.title('threshold vs dice')
    plt.savefig(str(Path(str(Out_Path), 'threshold_dice.png')), format='png', dpi=200)

    dice_all_final = []
    for filename in Path(test_path, 'images').rglob('*.png'):
        img = io.imread(filename, as_gray=True)
        img = img / 255
        img_pad = cv2.copyMakeBorder(img, 0, (PATCH_SIZE - img.shape[0]) % PATCH_SIZE, 0,
                                     (PATCH_SIZE - img.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
        p_img, i_img = Amir_utils.extract_grayscale_patches(img_pad, (PATCH_SIZE, PATCH_SIZE),
                                                            stride=(PATCH_SIZE, PATCH_SIZE))
        p_img = np.reshape(p_img, p_img.shape + (1,))
        p_img_predicted = model.predict(p_img)

        p_img_predicted = np.reshape(p_img_predicted, p_img_predicted.shape[:-1])
        img_mask_predicted_recons = Amir_utils.reconstruct_from_grayscale_patches(p_img_predicted, i_img)[0]

        # unpad and normalize
        img_mask_predicted_recons_unpad = img_mask_predicted_recons[0:img.shape[0], 0:img.shape[1]]
        img_mask_predicted_recons_unpad_norm = cv2.normalize(src=img_mask_predicted_recons_unpad, dst=None, alpha=0,
                                                             beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # quantization to make the binary masks

        img_mask_predicted_recons_unpad_norm[img_mask_predicted_recons_unpad_norm < best_threshold * 255] = 0
        img_mask_predicted_recons_unpad_norm[img_mask_predicted_recons_unpad_norm >= best_threshold * 255] = 255

        io.imsave(Path(str(Out_Path), Path(filename).name), img_mask_predicted_recons_unpad_norm)

        # DICE
        gt_path = str(Path(test_path, 'masks_lines'))
        gt_name = filename.name.partition('.')[0] + '_front.png'
        gt = io.imread(str(Path(gt_path, gt_name)), as_gray=True)

        pix_acc, iou, dice, mcc_val = metrics_binary(img_mask_predicted_recons_unpad_norm, gt)
        soft_acc, soft_iou, soft_dice, soft_mcc = soft_metrics(img_mask_predicted_recons_unpad_norm, gt)

        pixacc_all.append(pix_acc)
        iou_all.append(iou)
        dice_all_final.append(dice)
        mcc_all.append(mcc_val)

        soft_acc_all.append(soft_acc)
        soft_iou_all.append(soft_iou)
        soft_dice_all.append(soft_dice)
        soft_mcc_all.append(soft_mcc)

        DICE_all.append(distance.dice(gt.flatten(), img_mask_predicted_recons_unpad_norm.flatten()))
        EUCL_all.append(distance.euclidean(gt.flatten(), img_mask_predicted_recons_unpad_norm.flatten()))
        test_file_names.append(filename.name)

    DICE_avg = np.mean(DICE_all)
    EUCL_avg = np.mean(EUCL_all)
    pixacc_all_avg = np.mean(pixacc_all)
    iou_all_avg = np.mean(iou_all)
    dice_all_avg = np.mean(dice_all_final)
    mcc_avg = np.mean(mcc_all)
    soft_acc_avg = np.mean(soft_acc_all)
    soft_iou_avg = np.mean(soft_iou_all)
    soft_dice_avg = np.mean(soft_dice_all)
    soft_mcc_avg = np.mean(soft_mcc_all)

    Perf['DICE_all'] = DICE_all
    Perf['DICE_avg'] = DICE_avg
    Perf['EUCL_all'] = EUCL_all
    Perf['EUCL_avg'] = EUCL_avg
    Perf['pixacc_all'] = pixacc_all
    Perf['pixacc_all_avg'] = pixacc_all_avg
    Perf['iou_all'] = iou_all
    Perf['iou_all_avg'] = iou_all_avg
    Perf['dice_all'] = dice_all_final
    Perf['dice_all_avg'] = dice_all_avg
    Perf['test_file_names'] = test_file_names

    Perf['soft_acc_avg'] = soft_acc_avg
    Perf['soft_iou_avg'] = soft_iou_avg
    Perf['soft_dice_avg'] = soft_dice_avg
    Perf['soft_mcc_avg'] = soft_mcc_avg

    np.savez(str(Path(str(Out_Path), 'Performance.npz')), Perf)

    for filename in Path(Out_Path).rglob('*.png'):
        predicted_mask = io.imread(filename, as_gray=True)
        test_gtzones_path = Path('data_' + str(PATCH_SIZE) + pre_name + '/test/masks_zones')

    with open(str(Path(str(Out_Path), 'ReportOnModel.txt')), 'a') as f:
        f.write(str(Perf['DICE_avg']) + ', ' + str(Perf['EUCL_avg']) + '\n')
        f.write('loss used = ' + lossname + '\n')
        f.write('pixel accuracy = ' + str(Perf['pixacc_all_avg']) + '\n')
        f.write('Intersection over union = ' + str(Perf['iou_all_avg']) + '\n')
        f.write('dice coeff = ' + str(Perf['dice_all_avg']) + '\n')
        f.write('mcc = ' + str(mcc_avg) + '\n')
        f.write('soft_acc = ' + str(Perf['soft_acc_avg']) + '\n')
        f.write('soft_iou = ' + str(Perf['soft_iou_avg']) + '\n')
        f.write('soft_dice = ' + str(Perf['soft_dice_avg']) + '\n')
        f.write('soft_mcc = ' + str(Perf['soft_mcc_avg']) + '\n')
        f.close()

END = time.time()
print('Execution Time: ', END - START)