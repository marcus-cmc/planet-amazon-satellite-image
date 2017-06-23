# -*- coding: utf-8 -*-
# @Author: marcus
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from util_func import N_LABELS, N_CLOUDS
from util_func import load_image, encode_tags
from keras.preprocessing.image import ImageDataGenerator


class BatchImgGen(object):
    """
    A custom batch image data / label generator.
    Can be called infinite number of times.
    """
    def __init__(self, data, batch_size=64, pixel=None, aug_times=0):
        """
        aug_times:
            num of batches of augmented images (in addition to original image)
            to generate. if aug_times = 2 and batch_size=64, then each batch
            would have 192 images (64 original, 128 augmented)
        """
        self.df = data  # a pd dataframe
        self.idx = np.arange(len(data))
        self.curr_idx = 0
        self.batch_size = batch_size
        self.pixel = pixel
        self.steps = self._cal_steps()
        self.aug_times = aug_times  # 0: no image augumentation

    def get_steps(self):
        return self.steps

    def _cal_steps(self):
        steps, remain = divmod(len(self.df), self.batch_size)
        return steps + (remain != 0) * 1

    def __next__(self):
        return self._get_batch()

    def next(self):
        return self.__next__()

    def _get_batch(self):
        pix = self.pixel or 256
        n_imgs = self.batch_size * (1 + self.aug_times)
        X = np.zeros((n_imgs, pix, pix, 3), dtype=np.float32)
        Y = np.zeros((n_imgs, N_LABELS), dtype=np.uint8)
        for i in range(self.batch_size):
            index = self.idx[self.curr_idx]
            X[i] = self._get_img(index)
            Y[i] = self._get_label(index)
            self._update_curr_idx()

        if self.aug_times:   # image augmentation
            X_orig, Y_orig = X[:self.batch_size], Y[:self.batch_size]
            keras_aug = ImageDataGenerator(
                horizontal_flip=True, vertical_flip=True,
                rotation_range=15.0, channel_shift_range=0.05,
                fill_mode='nearest', data_format="channels_last")

            aug = keras_aug.flow(X_orig, Y_orig, batch_size=self.batch_size)
            for aug_batch in range(1, 1+self.aug_times):
                aug_x, aug_y = next(aug)
                lo = aug_batch * self.batch_size
                X[lo:lo+self.batch_size] = aug_x
                Y[lo:lo+self.batch_size] = aug_y

        return X, Y

    def _get_img(self, index):
        return load_image(self.df.image_name.values[index], self.pixel)

    def _get_label(self, index):
        return encode_tags(self.df.tags.values[index])

    def _update_curr_idx(self):
        self.curr_idx += 1
        if self.curr_idx == len(self.idx):
            self._shuffle_idx()
            self.curr_idx = 0
        return

    def _shuffle_idx(self):
        np.random.shuffle(self.idx)  # not the index in df
        return


class TestImgIter(BatchImgGen):
    def __iter__(self):
        return self

    def _get_batch(self):
        if self.curr_idx == len(self.df):
            raise StopIteration
        pix = self.pixel or 256
        n_samples = min(self.batch_size, len(self.df) - self.curr_idx)
        X = np.zeros((n_samples, pix, pix, 3), dtype=np.float32)
        for i in range(n_samples):
            X[i] = self._get_img(self.curr_idx)
            self.curr_idx += 1
        return X


class BatchImgGen2(BatchImgGen):
    """For PlanetAmazonCNN2 (two outputs, cloud and common labels)"""

    def __next__(self):
        X, Y = self._get_batch()
        return X, {"cloud_output": Y[:, :N_CLOUDS],
                   "common_output": Y[:, N_CLOUDS:]}
