# -*- coding: utf-8 -*-
# @Author: marcus

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import numpy as np
import pandas as pd
import collections
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import backend as K
from keras.callbacks import History, ModelCheckpoint
from util_func import LABEL_NAMES, LABEL_MAPPING, N_LABELS, N_CLOUDS, N_COMMON
from util_func import load_image, encode_tags, decode_tags
from util_func import find_thresholds, bin_by_thresholds, decode_and_save
from batch_img_gen import BatchImgGen, BatchImgGen2, TestImgIter

CHANNELS = 3


class PlanetAmazonCNN(object):
    def __init__(self, modelname, pixel, aug_times=0, optimizer=None,
                 callbacks=True, drop_dense=0.3, drop_pooling=0.3,
                 norm_input=True, norm_conv=True,
                 norm_pooling=True, norm_dense=True):

        self.modelname = modelname
        self.pixel = pixel
        self.aug_times = aug_times
        self.epoch = 0
        if optimizer is None:
            self.optimizer = optimizers.Adam()
        self.drop_dense = drop_dense
        self.drop_pooling = drop_pooling
        self.history = collections.defaultdict(list)
        self.norm_input = norm_input
        self.input_layer = Input(shape=(pixel, pixel, CHANNELS))
        self.stack = self.input_layer
        if norm_input:
            self.stack = self.add_batch_norm(self.stack)
        self.path = self._path()
        self.callbacks = [self._add_checkpoint()] if callbacks else []
        # the following only affect "add_conv2d_block" and "add_dense_block"
        self.norm_conv = norm_conv
        self.norm_pooling = norm_pooling
        self.norm_dense = norm_dense

    def _add_checkpoint(self):
        fpath = ("ep{epoch:02d}_loss{val_loss:.3f}_" +
                 "acc{val_acc:.3f}.hdf5")
        fpath = os.path.join(self.path, fpath)

        return ModelCheckpoint(fpath, monitor='val_loss', verbose=0,
                               save_best_only=True, save_weights_only=False,
                               mode='min', period=1)

    def _path(self):
        s = "".join(c if c.isalnum() or c in "-.()" else "_ "
                    for c in self.modelname)
        s = "_".join(c for c in s.split("_ ") if c)
        path = os.path.join(os.getcwd(), "logs", s)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _make_default_CNN(self):
        L = self.stack
        L = self.add_conv2d_block(L, (32, 64))
        L = self.add_conv2d_block(L, (64, 128))
        L = self.add_conv2d_block(L, (128, 256))
        L = self.add_conv2d_block(L, (256, 256))
        L = self.flatten(L)
        return L

    def make_default_model(self):
        flat = self._make_default_CNN()
        L = self.add_dense_block(flat, (128,))
        self.add_output(L)
        self.compile_model()

    def compile_model(self):
        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)
        self.model.compile(optimizer=self.optimizer,
                           loss="binary_crossentropy",
                           metrics=["accuracy"])

    def add_conv2d_block(self, L, filters):
        for filter_size in filters:
            L = self.add_conv2d(L, filter_size)
            if self.norm_conv:
                L = self.add_batch_norm(L)
        L = self.add_pooling(L)
        if self.norm_pooling:
            L = self.add_batch_norm(L)
        L = self.add_dropout(L, self.drop_pooling)

        return L

    def add_dense_block(self, L, units):
        for unit in units:
            L = self.add_dense(L, unit)
            if self.norm_dense:
                L = self.add_batch_norm(L)
            L = self.add_dropout(L, self.drop_dense)
        return L

    def add_conv2d(self, L, filter_size, kernel_size=(3, 3)):
        return Conv2D(filter_size, kernel_size=kernel_size,
                      activation="relu", padding='same',
                      data_format="channels_last")(L)

    def add_pooling(self, L, pool_size=(2, 2)):
        return MaxPooling2D(pool_size=pool_size,
                            data_format="channels_last")(L)

    def add_dense(self, L, units):
        return Dense(units, activation="relu")(L)

    def add_dropout(self, L, rate):
        return Dropout(rate=rate)(L) if rate else L

    def flatten(self, L):
        return Flatten()(L)

    def add_batch_norm(self, L):
        return BatchNormalization()(L)

    def add_output(self, L):
        self.output_layer = Dense(units=N_LABELS, activation="sigmoid")(L)
        return self.output_layer

    def get_learning_rate(self):
        return K.eval(self.model.optimizer.lr)

    def set_learning_rate(self, learning_rate):
        K.set_value(self.optimizer.lr, learning_rate)
        return

    def get_train_data_gen(self):
        return BatchImgGen(data=self.train_data, batch_size=self.batch_size,
                           pixel=self.pixel, aug_times=self.aug_times)

    def _format_validation_data(self, validation_data):
        return validation_data

    def fit_generator(self, data_df, batch_size=64, steps_per_epoch=None,
                      validation_data=None, epochs=1, **kwargs):
        # to do: make find_thresholds optional
        self.batch_size = batch_size
        self.train_data = data_df
        self.validation_data = self._format_validation_data(validation_data)
        self.train_data_gen = self.get_train_data_gen()
        self._fit_generator(epochs=epochs, **kwargs)

    def fit(self, *args, **kwargs):
        epochs = kwargs.get('epochs', 1)
        kwargs['epochs'] = epochs + self.epochs
        kwargs['initial_epoch'] = self.epoch
        kwargs['callbacks'] = self.callbacks
        h = self.model.fit(*args, **kwargs)
        self.epoch += epochs
        self._update_history(h)
        if self.validation_data is not None:
            self.thresholds = self._find_thresholds()
        return h

    def _fit_generator(self, epochs, steps_per_epoch=None, **kwargs):
        if steps_per_epoch is None:
            steps_per_epoch = self.train_data_gen.get_steps()
        h = self.model.fit_generator(self.train_data_gen,
                                     steps_per_epoch=steps_per_epoch,
                                     validation_data=self.validation_data,
                                     epochs=self.epoch+epochs,
                                     initial_epoch=self.epoch,
                                     callbacks=self.callbacks,
                                     **kwargs)
        self._update_history(h)
        self.epoch += epochs
        if self.validation_data is not None:
            self.thresholds = self._find_thresholds()

    def resume_fit(self, epochs=1, lr=None, steps_per_epoch=None):
        if lr is not None:
            self.set_learning_rate(lr)
        self._fit_generator(epochs, steps_per_epoch)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def predict_testset(self, data_test_df, test_batch_size=256):
        self.test_data_iter = TestImgIter(data=data_test_df,
                                          batch_size=test_batch_size,
                                          pixel=self.pixel)

        Y_pred = np.zeros((len(data_test_df), N_LABELS))

        for batch, test_imgs in enumerate(self.test_data_iter):
            idx = batch * test_batch_size
            if batch % 10 == 0:
                print(" batch {}:\t {}/{}\t images".format(batch, idx,
                                                           len(data_test_df)))
            Y_pred[idx: idx+test_batch_size] = self.predict(test_imgs,
                                                            verbose=0)
        self.Y_pred = Y_pred
        self.Y_pred_label = self._bin_Y_pred()
        self.save_results()

        return self.Y_pred

    def _update_history(self, h):
        for k in h.history:
            pad = [0] * (self.epoch - len(self.history[k]))
            self.history[k].extend(pad + h.history[k])
        df = pd.DataFrame(self.history)
        df.index.name = "epoch"
        df.to_csv(os.path.join(self.path, "Model_history.csv"))

    def load_prev_model(self, epoch):
        pass

    def _bin_Y_pred(self):
        if self.validation_data is not None:
            self.thresholds = self._find_thresholds()
            return bin_by_thresholds(self.Y_pred, self.thresholds)

        return bin_by_thresholds(self.Y_pred)

    def _save_thresholds(self, thresholds):
        prefix = "val_pred_ep{:02d}".format(self.epoch-1)
        prefix = os.path.join(self.path, prefix)
        df = pd.DataFrame()
        df["label_name"] = LABEL_NAMES
        df["thresholds"] = thresholds
        df.to_csv(prefix + "_thresholds.csv", index=False)
        return

    def _find_thresholds(self):
        X_val, Y_val = self.validation_data
        Y_val_pred = self.model.predict(X_val)
        thresholds, Y_val_pred_label = find_thresholds(Y_val, Y_val_pred)
        self._save_valid_pred(Y_val_pred, Y_val_pred_label)
        self._save_thresholds(thresholds)
        return thresholds

    def _save_valid_pred(self, Y_val_pred, Y_val_pred_label):
        prefix = "val_pred_ep{:02d}".format(self.epoch-1)
        prefix = os.path.join(self.path, prefix)
        df_label = pd.DataFrame(Y_val_pred_label, columns=LABEL_NAMES)
        df_pred = pd.DataFrame(Y_val_pred, columns=LABEL_NAMES)
        df_pred.to_csv(prefix + "_probability.csv")
        df_label.to_csv(prefix + "_label.csv")
        return

    def save_valid_df(self, df_validation):
        df_validation.to_csv(os.path.join(self.path, "validation_label.csv"),
                             index=False)

    def save_results(self, savename=""):
        self.save_summary(savename)

        if savename:  # epoch-1 to make it consistent with callbacks
            prefix = "{}_ep{:02d}".format(savename, self.epoch-1)
        else:
            prefix = "ep{:02d}".format(self.epoch-1)

        prefix = os.path.join(self.path, prefix)

        df_prob = pd.DataFrame(self.Y_pred, columns=LABEL_NAMES)
        df_prob.index.name = self.test_data_iter.df.image_name
        df_prob.to_csv(prefix + "_probability.csv")

        return decode_and_save(self.test_data_iter.df, self.Y_pred_label,
                               prefix)

    def save_summary(self, fname=""):
        """ workaround to save model.summary() to a .txt file"""
        original_stdout = sys.stdout
        fname = os.path.join(self.path, fname + "summary.txt")

        with open(fname, 'w') as file:
            sys.stdout = file
            print(self.model.summary())
            attrs = self.__dict__
            for k in sorted(attrs.keys()):
                if type(attrs[k]) in (int, float, bool):
                    file.write("{}:\t{}\n".format(k, attrs[k]))
        sys.stdout = original_stdout

        return


class PlanetAmazonCNN2(PlanetAmazonCNN):
    """
    Two different output layers with shared convolutional NNs
        cloud_output: softmax (4 labels, choose only one)
        common_output: sigmoid (13 labels)
    """

    def _add_checkpoint(self):
        fpath = ("ep{epoch:02d}_val_loss{val_loss:.3f}_" +
                 "val_cloud_loss{val_cloud_output_loss:.3f}.hdf5")
        fpath = os.path.join(self.path, fpath)

        return ModelCheckpoint(fpath, monitor='val_loss', mode='min',
                               save_best_only=True, save_weights_only=False,
                               verbose=0, period=1)

    def make_default_model(self):
        flat = self._make_default_CNN()
        L_clouds = self.add_dense_block(flat, (64,))
        L_common = self.add_dense_block(flat, (128,))
        self.add_output(L_clouds, L_common)
        self.compile_model()

    def compile_model(self):
        self.model = Model(inputs=self.input_layer,
                           outputs=[self.output_cloud, self.output_common])

        self.model.compile(optimizer=self.optimizer,
                           loss={"cloud_output": "categorical_crossentropy",
                                 "common_output": "binary_crossentropy"},
                           metrics=["accuracy"])
        return

    def add_output(self, L_clouds, L_common):
        self.output_cloud = Dense(units=N_CLOUDS, activation="softmax",
                                  name="cloud_output")(L_clouds)

        self.output_common = Dense(units=N_COMMON, activation="sigmoid",
                                   name="common_output")(L_common)
        return

    def get_train_data_gen(self):
        return BatchImgGen2(data=self.train_data, batch_size=self.batch_size,
                            pixel=self.pixel, aug_times=self.aug_times)

    def _format_validation_data(self, validation_data):
        if validation_data is not None:
            X, Y = validation_data
            return X, {"cloud_output": Y[:, :N_CLOUDS],
                       "common_output": Y[:, N_CLOUDS:]}
        return

    def predict(self, *args, **kwargs):
        Y_pred = self.model.predict(*args, **kwargs)
        Y_pred = np.concatenate(Y_pred, axis=1)
        return Y_pred

    def _bin_Y_pred(self):
        if self.validation_data is not None:
            self.thresholds = self._find_thresholds()
            return bin_by_thresholds(self.Y_pred, self.thresholds,
                                     softmax_clouds=True)

        return bin_by_thresholds(self.Y_pred, softmax_clouds=True)

    def _find_thresholds(self):
        X_val, Y_val = self.validation_data
        Y_val = np.concatenate((Y_val["cloud_output"],
                                Y_val["common_output"]), axis=1)
        Y_val_pred = self.predict(X_val)

        thresholds, Y_val_pred_label = find_thresholds(Y_val, Y_val_pred,
                                                       softmax_clouds=True)
        self._save_valid_pred(Y_val_pred, Y_val_pred_label)
        self._save_thresholds(thresholds)

        return thresholds
