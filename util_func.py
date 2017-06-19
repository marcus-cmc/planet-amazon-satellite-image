# -*- coding: utf-8 -*-
# @Author: marcus

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
from sklearn.metrics import fbeta_score


LABEL_CLOUDS = ['cloudy', 'partly_cloudy', 'haze', 'clear']
LABEL_COMMON = [
    'agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down',
    'conventional_mine', 'cultivation', 'habitation', 'primary', 'road',
    'selective_logging', 'slash_burn', 'water']

LABEL_NAMES = LABEL_CLOUDS + LABEL_COMMON
LABEL_MAPPING = {tag: i for i, tag in enumerate(LABEL_NAMES)}

N_CLOUDS = len(LABEL_CLOUDS)
N_COMMON = len(LABEL_COMMON)
N_LABELS = len(LABEL_NAMES)

# IMG_PATH = {"train": "./data/train-jpg/",
#             "test": "./data/test-jpg/",
#             "file": "./data/test-jpg-additional/"}

"""
main directory/
    util_func.py
    planet_amazon_clf.py
    batch_img_gen.py
    data/
        train-jpg/
        test-jpg/
        test-jpg-additional/
    logs/
"""
IMG_PATH = {"train": os.path.join(os.getcwd(), "data", "train-jpg"),
            "test": os.path.join(os.getcwd(), "data", "test-jpg"),
            "file": os.path.join(os.getcwd(), "data", "test-jpg-additional")}


def load_image(img_name, pixel_per_side=None):
    """ load and resize (if needed) images from disk"""
    path = IMG_PATH[img_name.split("_")[0]]
    img_file = os.path.join(path, img_name + ".jpg")
    img = skimage.io.imread(img_file)
    if pixel_per_side is not None:
        img = skimage.transform.resize(img, (pixel_per_side, pixel_per_side),
                                       preserve_range=True, mode='constant')
    return img / 255.0  # np-array, shape (256, 256, 3)


def encode_tags(tags):
    """
    encode tags from an image into a binary list according to LABEL_MAPPING.
    length: N_LABELS, the first 4 are clouds, and the rest 13 are common labels
    """
    res = [0] * N_LABELS
    for name in tags.split(" "):
        res[LABEL_MAPPING[name]] = 1
    return res


def decode_tags(encoded):
    """decode a binary array/list into the desired output format"""
    return " ".join(LABEL_NAMES[i] for i, val in enumerate(encoded) if val)


def get_imgs_and_labels(data_df, pixel_per_side=None):
    """
    get both images and labels from data_df
    data_df must be (a slice) from "train_v2.csv" from kaggle
    (no labels for test data)
    """
    return get_imgs(data_df, pixel_per_side), get_labels(data_df)


def get_imgs(data_df, pixel_per_side=None):
    """
    get images as a numpy array
    aarguments:
        data_df: a DataFrame from "train_v2.csv" or "sample_submission_v2.csv"
                 from kaggle (or a slice of it)
        pixel_per_side: the size of returned image, (if None, no resizing)
    return:
        a np array of shape : (len(data_df), pixel_per_side, pixel_per_side, 3)
    """
    pix = pixel_per_side or 256
    # pix = pixel_per_side if pixel_per_side is not None else 256
    X = np.zeros((len(data_df), pix, pix, 3), dtype=np.float32)
    for i, img_name in enumerate(data_df.image_name.values):
        X[i] = load_image(img_name, pixel_per_side)
    return X


def get_labels(data_df):
    """
    get labels of images as a numpy array
    aarguments:
        data_df: a DataFrame from "train_v2.csv" or "sample_submission_v2.csv"
                 from kaggle (or a slice of it)
    return:
        a np array of shape: (len(data_df), N_LABELS)
    """
    Y = np.zeros((len(data_df), N_LABELS), dtype=np.uint8)
    for i, tags in enumerate(data_df.tags.values):
        Y[i] = encode_tags(tags)
    return Y


def bin_clouds(Y_pred):
    """
    binarize the prediction for clouds labels, only used in PlanetAmazonCNN2
    argument:
        Y_pred: a np array from the prediction results from PlanetAmazonCNN2
                shape: (n_samples, N_LABELS)
    """
    Y_label = np.copy(Y_pred)
    cloud_max = np.amax(Y_pred[:, :N_CLOUDS], axis=1)
    Y_label[:, :N_CLOUDS] = (
        np.where(Y_pred[:, :N_CLOUDS] == cloud_max[:, None], 1, 0))

    return Y_label


def bin_by_thresholds(Y_pred, thresholds=None, softmax_clouds=False):
    """
    binarize the prediction according to the given thresholds
    argument:
        Y_pred: a np array from the prediction results
        thresholds: an iterable of length N_LABELS
        softmax_clouds: False for PlanetAmazonCNN; True for PlanetAmazonCNN2
    returns: a np array of shape (n_samples, N_LABELS)
    """
    if softmax_clouds:
        Y_label, start_idx = bin_clouds(Y_pred), N_CLOUDS
    else:
        Y_label, start_idx = np.copy(Y_pred), 0

    if thresholds is None:
        thresholds = np.array([0.175] * N_LABELS)

    for i in range(start_idx, N_LABELS):
        Y_label[:, i] = 1 * (Y_pred[:, i] >= thresholds[i])

    return Y_label


def find_thresholds(actual, Y_pred, softmax_clouds=False, base=0.175,
                    print_f=True, th_range=np.arange(0.03, 0.501, 0.002)):
    """
    use actual data to find the thresholds for each label for a given model
    arguments:
        actual: actual labels, a (n_samples, N_LABELS) binary np array
                could be from training or validation data
        Y_pred: predicted label, a (n_samples, N_LABELS) np array (floats)
        base: a float between 0 and 1 used to binarize the prediction initially
        th_range: thresold candidates to try, an iterable
                  each element is between 0 and 1
    return:
        best thresholds according to this input, a (N_LABELS, ) np array
        Y_label
    """
    if softmax_clouds:
        Y_label, start_idx = bin_clouds(Y_pred), N_CLOUDS
        Y_label[:, N_CLOUDS:] = (Y_pred[:, N_CLOUDS:] >= base)
        thresholds = [1] * N_CLOUDS  # pad with 1
    else:
        Y_label, start_idx = Y_pred >= base, 0
        thresholds = []

    f_max = 0
    for i in range(start_idx, N_LABELS):
        score = []
        for j, th in enumerate(th_range):
            Y_label[:, i] = Y_pred[:, i] >= th
            score.append(fbeta_score(actual, Y_label, beta=2,
                         average='samples'))
        f_max = max(f_max, max(score))
        best_th = th_range[np.argmax(score)]
        thresholds.append(best_th)
        Y_label[:, i] = Y_pred[:, i] >= best_th

    if print_f:
        print("best_f2score: {}".format(f_max))

    return np.array(thresholds), 1 * Y_label

def find_thresholds2(actual, Y_pred, softmax_clouds=False, base=0.175, k=3,
                     print_f=True, th_range=np.arange(0.03, 0.501, 0.001)):

    if softmax_clouds:
        Y_label, start_idx = bin_clouds(Y_pred), N_CLOUDS
        Y_label[:, N_CLOUDS:] = (Y_pred[:, N_CLOUDS:] >= base)
        thresholds = [1] * N_CLOUDS  # pad with 1
    else:
        Y_label, start_idx = Y_pred >= base, 0
        thresholds = []

    f_max = 0
    n = len(Y_pred)
    for fold in range(k):
        th_fold = [1] * N_CLOUDS if softmax_clouds else []
        lo, hi = fold * n // k, (fold+1) * n // k
        for i in range(start_idx, N_LABELS):
            score = []
            for j, th in enumerate(th_range):
                Y_label[lo:hi, i] = Y_pred[lo:hi, i] >= th
                score.append(fbeta_score(actual[lo:hi], Y_label[lo:hi], beta=2,
                                         average='samples'))
            f_max = max(f_max, max(score))
            best_th = th_range[np.argmax(score)]
            th_fold.append(best_th)
            Y_label[lo:hi, i] = Y_pred[lo:hi, i] >= best_th
        thresholds.append(th_fold)
    thresholds = np.mean(thresholds, axis=0)

    Y_label = np.where(Y_pred >= thresholds, 1, 0)
    best_f = fbeta_score(actual, Y_label, beta=2, average='samples')

    if print_f:
        print("best_f2score: {}".format(best_f))

    return np.array(thresholds), 1*Y_label

def decode_and_save(data_test_df, Y_label, savename=""):
    """save the predited labels as the desired format for submission"""
    df = pd.DataFrame()
    df["image_name"] = data_test_df.image_name
    df["tags"] = [decode_tags(tags) for tags in Y_label]
    df.to_csv(savename + "_prediction.csv", index=False)

    return df
