# -*- coding: utf-8 -*-
# @Author: marcus

import os
import importlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")


from util_func2 import LABEL_NAMES, LABEL_MAPPING, N_LABELS, N_CLOUDS, N_COMMON
from util_func2 import load_image, encode_tags, decode_tags
from util_func2 import find_thresholds, bin_by_thresholds, decode_and_save
from util_func2 import get_imgs, get_labels, get_imgs_and_labels
from batch_img_gen2 import BatchImgGen, TestImgIter
from planet_amazon_clf2 import PlanetAmazonCNN


PIX = 64

TRAIN = pd.read_csv("./data/train_v2.csv")   # length: 40479
TEST = pd.read_csv("./data/sample_submission_v2.csv")  # length 61191
df_train, df_validation = train_test_split(TRAIN, train_size=0.05,
                                           random_state=3)

X_valid, Y_valid = get_imgs_and_labels(df_validation[:100], PIX)
validation_data = X_valid, {"cloud_output": Y_valid[:, :N_CLOUDS],
                            "common_output": Y_valid[:, N_CLOUDS:]}

PA = PlanetAmazonCNN("test_model", PIX)
PA.make_default_model()
PA.model.summary()

PA.fit_generator(df_train[:200], batch_size=64,
                 validation_data=validation_data, epochs=2)

PA.predict_testset(TEST[:100], test_batch_size=70)
r = PA.save_results()





