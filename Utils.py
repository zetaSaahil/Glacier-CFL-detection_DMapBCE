
import keras
import numpy as np
from keras import backend as K

class custom_earlystop(keras.callbacks.Callback):
    def __init__(self, monitor = 'val_acc', previous_historylog= None, prev_bestweights = None, patience=0):
        super(custom_earlystop, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.previous = previous_historylog
        self.store = None
        self.best_weights = None
        self.prev_weights = prev_bestweights

    def on_train_begin(self, logs=None):
        if self.previous is not None:
            self.store = self.previous
            self.best_weights = self.prev_weights
        else:
            self.store = np.array([])


    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        print(current)
        self.store = np.append(self.store, current)

        if self.monitor == 'val_acc' or self.monitor == 'val_mcc':
            max_val = np.max(self.store)
            index = np.argmax(self.store)
            current_index = self.store.size - 1
            if current_index - index >= self.patience:
                self.model.set_weights(self.best_weights)
                self.model.stop_training = True
            if current >= max_val:
                self.best_weights = self.model.get_weights()
            else:
                self.best_weights = self.best_weights

        if self.monitor == 'val_loss':
            min_val = np.min(self.store)
            index = np.argmin(self.store)
            current_index = self.store.size - 1
            if current_index - index >= self.patience:
                self.model.set_weights(self.best_weights)
                self.model.stop_training = True
            if current <= min_val:
                self.best_weights = self.model.get_weights()
            else:
                self.best_weights = self.best_weights

        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return

class CyclicLR(keras.callbacks.Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency.
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.
    # Example for CIFAR-10 w/ batch size 100:
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # References
      - [Cyclical Learning Rates for Training Neural Networks](
      https://arxiv.org/abs/1506.01186)
    """

    def __init__(
            self,
            base_lr=0.001,
            max_lr=0.006,
            step_size=2000.,
            mode='triangular',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault(
            'lr', []).append(
            K.get_value(
                self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

from preprocessing import Filter
import cv2
from scipy.ndimage import median_filter

filter = Filter()
def apply_filter(name, data):
    if name == 'lee':
        data = filter.lee_filter(data, 5)
        #data = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25)).apply((data).astype('uint8'))  # CLAHE adaptive contrast enhancement
    if name == 'elee':
        data = filter.enhanced_lee(data)
        #data = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25)).apply((data).astype('uint8'))  # CLAHE adaptive contrast enhancement
    if name == 'kuan':
        data = filter.kuan_filter(data,11,8)
        data = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25)).apply((data).astype('uint8'))  # CLAHE adaptive contrast enhancement
    if name == 'median':
        data = median_filter(data,5)
        #data = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25)).apply((data).astype('uint8'))  # CLAHE adaptive contrast enhancement
    if name == 'leeclahe':
        data = filter.lee_filter(data, 5)
        data = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25)).apply((data).astype('uint8'))  # CLAHE adaptive contrast enhancement
    if name == 'medianclahe':
        data = median_filter(data, 5)
        data = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(20, 20)).apply((data).astype('uint8'))  # CLAHE adaptive contrast enhancement
    else:
        data = data
    return data

#remove bad quality factor images
def remove_images(dataframe, quality_factor):
    assert isinstance(quality_factor, int)
    newdataframe = dataframe
    for i in dataframe.index:
        imagename = dataframe.loc[i, 'images']
        imagename_wtype = imagename.split('.')[2]
        qf_image = int(imagename_wtype.split('_')[4])
        if qf_image == quality_factor:
            newdataframe = newdataframe.drop([i])

    return newdataframe