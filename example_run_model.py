# -*- coding: utf-8 -*-
# @Author: C. Marcus Chuang

import pandas as pd
from sklearn.model_selection import train_test_split
from util_func import get_imgs_and_labels
from planet_amazon_clf import PlanetAmazonCNN

"""
Example script, public LB score ~0.945
"""

PIX = 128

TRAIN = pd.read_csv("./data/train_v2.csv")   # length: 40479
TEST = pd.read_csv("./data/sample_submission_v2.csv")  # length 61191
df_train, df_validation = train_test_split(TRAIN, train_size=0.8,
                                           random_state=3)
validation_data = get_imgs_and_labels(df_validation, PIX)

clf = PlanetAmazonCNN("Exanple_model", PIX,
                      drop_dense=0.3, drop_pooling=0.0,
                      norm_conv=False, aug_times=2)

clf.make_default_model()
clf.compile_model()
clf.model.summary()

clf.fit_generator(df_train, batch_size=64, epochs=8,
                  validation_data=validation_data)
clf.resume_fit(epochs=4, lr=5E-4)
clf.resume_fit(epochs=2, lr=2.5E-4)
clf.resume_fit(epochs=2, lr=1E-4)

clf.predict_testset(TEST, test_batch_size=512)
