from keras import backend as K
import tensorflow as tf
from scipy.ndimage.morphology import distance_transform_edt as edt
import cv2
import numpy as np

def focal_loss(y_true, y_pred):
    gamma = 2.
    alpha = .33
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


'''
def focal_loss(y_true, y_pred):
    gamma = 2.
    alpha = .25
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.math.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)
'''
def dice_loss(y_true, y_pred):
    numerator = 2 * K.sum(y_true * y_pred)
    denominator = K.sum(y_true) + K.sum(y_pred)

    return 1 - (numerator + 1) / (denominator + 1)

def mcc_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float')) + K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'))
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float')) + K.sum(K.cast(y_true * y_pred, 'float'))
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float')) + K.sum(K.cast(y_true * (1 - y_pred), 'float'))
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float')) + K.sum(K.cast((1 - y_true) * y_pred, 'float'))

    fp = fp/100
    fn = fn/ 100

    up = tp * tn - fp * fn
    down = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    mcc = up / (down + K.epsilon())

    return 1 - (mcc)

# mcc loss approx 1
def mcc_loss1(eta = 1e-4):
    def mcc_loss_fixed(y_true, y_pred):
        tp = K.sum(K.cast(y_true * y_pred, 'float'))*(1 - eta) + K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'))*(eta)
        tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'))*(1 - eta) + K.sum(K.cast(y_true * y_pred, 'float'))*(eta)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'))*(1 - eta) + K.sum(K.cast(y_true * (1 - y_pred), 'float'))*(eta)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'))*(1 - eta) + K.sum(K.cast((1 - y_true) * y_pred, 'float'))*(eta)


        up = tp * tn - fp * fn
        down = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        mcc = up / (down + K.epsilon())

        return 1 - (mcc)
    return mcc_loss_fixed

# mcc loss approx 2
def mcc_loss2(eta = 0.9):
    def mcc_loss_fixed(y_true, y_pred):
        tp = K.sum(K.cast(y_true * y_pred, 'float'))
        tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float')) * eta
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'))
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float')) * eta

        up = tp * tn - fp * fn
        down = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        mcc = up / (down + K.epsilon())

        return 1 - (mcc)
    return mcc_loss_fixed

# original mcc loss
def mcc_loss3(eta = 100):
    def mcc_loss_fixed(y_true, y_pred):
        tp = K.sum(K.cast(y_true * y_pred, 'float'))
        tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'))
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'))
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'))

        up = tp * tn - fp * fn
        down = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        mcc = up/ (down + K.epsilon())

        return 1 - K.mean(mcc)
    return mcc_loss_fixed

# bce with class weights
def weighted_bce(w = 99.):
    def bce_fixed(y_true, y_pred):
        weights = (y_true * w) + 1.
        bce = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = K.mean(bce * weights)
        return weighted_bce
    return bce_fixed

def bce_mcc_loss(wbce=0.7, wmcc=0.3):
    def bce_mcc_fixed(y_true, y_pred):
        bce = wbce * K.binary_crossentropy(y_true, y_pred)

        tp = K.sum(K.cast(y_true * y_pred, 'float'))
        tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'))
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'))
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'))

        up = tp * tn - fp * fn
        down = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        mcc = up / (down + K.epsilon())

        mcc_loss = wmcc*(1 - (mcc))

        return bce + mcc_loss
    return bce_mcc_fixed

# distance map calculation for dmap loss
def calc_dist_map(y,relax):
    f_size = 3 #paramter w of dmap loss
    k = 1      #parameter k of dmap loss
    y = np.squeeze(y, -1)
    if y.max() == 0:
        soft_gt = 1 - y
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (f_size, f_size))
        fat_gt = cv2.dilate(y, kernel, iterations=1)
        soft_gt = edt(fat_gt) / relax
        soft_gt = 1 / (1 + np.exp(-soft_gt))
        soft_gt = soft_gt - soft_gt.min()
        soft_gt = soft_gt / soft_gt.max()
        inv_gt = 1 - fat_gt
        soft_gt = soft_gt + inv_gt*k

    return soft_gt

# including dmaps in batches
def calc_dist_map_batch(y_true, relax):
    y_true_numpy = y_true.numpy()
    dist_y_true = np.array([calc_dist_map(y,relax)
                     for y in y_true_numpy]).astype(np.float32)
    dist_y_true = np.expand_dims(dist_y_true, -1)
    #print(dist_y_true.shape)
    return dist_y_true

def d_map_weighted_bce(w=99., relax=4):
    def d_map_weighted_bce_fixed(y_true, y_pred):
        y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                         inp=[y_true, relax],
                                         Tout=tf.float32)
        y_pred = y_pred*y_true_dist_map
        weights = (y_true * w) + 1.
        bce = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = K.mean(bce * weights)
        return weighted_bce

    return d_map_weighted_bce_fixed

def d_map_bce(relax=4):
    def d_map_bce_fixed(y_true, y_pred):
        y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                         inp=[y_true,relax],
                                         Tout=tf.float32)
        y_pred = y_pred * y_true_dist_map
        bce = K.binary_crossentropy(y_true, y_pred)
        return K.mean(bce)
    return d_map_bce_fixed

def d_map_mcc(relax =4):
    def d_map_mcc_fixed(y_true, y_pred):
        y_true = K.cast(y_true, 'float')
        y_pred = K.cast(y_pred, 'float')
        y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                         inp=[y_true, relax],
                                         Tout=tf.float32)
        y_pred = y_pred * y_true_dist_map

        tp = K.sum(K.cast(y_true * y_pred, 'float'))
        tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'))
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'))
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'))

        up = tp * tn - fp * fn
        down = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        mcc = up / (down + K.epsilon())

        return 1 - (mcc)
    return d_map_mcc_fixed