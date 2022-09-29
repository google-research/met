# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cProfile import label
import random
import math
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

num_rows_trn = 10000
num_rows_tst = 2000
num_cols_d = 5

def classification_label(x):
    if abs(x)<8.34:
        return 0
    return 1

def gaussian():
    orig_point = np.random.normal(0, 1)
    point = [orig_point]
    noise = []
    for i in range(num_cols_d*2-1):
        noise.append(np.random.normal(0,1))
        point.append(point[-1]*np.sqrt(0.5)+noise[-1]*np.sqrt(0.5))
    label = 0
    for _ in noise:
        label+=_**2
    return point, classification_label(label)

data = []
y = []

for i in range(num_rows_trn):
    row, lbl = gaussian()
    data.append(row)
    y.append(lbl)

data = np.array(data)
y = np.array(y)

clf = make_pipeline(StandardScaler(), LinearRegression())
clf.fit(data, y)


data_test = []
y_test = []

for i in range(num_rows_tst):
    row, lbl = gaussian()
    data_test.append(row)
    y_test.append(lbl)

data_test = np.array(data_test)
y_test  = np.array(y_test)

y_pred = clf.predict(data)
y_pred[y_pred>0.5]=1
y_pred[y_pred<=0.5]=0
y_pred_test = clf.predict(data_test)
y_pred_test[y_pred_test>0.5]=1
y_pred_test[y_pred_test<=0.5]=0

print("Train Acc, AUROC - Linear Model : ", accuracy_score(y_pred,y), roc_auc_score(y, clf.predict(data)))
print("Test Acc, AUROC - Linear Model : ", accuracy_score(y_pred_test,y_test), roc_auc_score(y_test, clf.predict(data_test)))

train_data = np.zeros((num_rows_trn, 2*num_cols_d+1))
train_data[:,0] = y
train_data[:,1:] = data

test_data = np.zeros((num_rows_tst, 2*num_cols_d+1))
test_data[:,0] = y_test
test_data[:,1:] = data_test

import pandas as pd
df = pd.DataFrame(train_data,columns=None)
df_test = pd.DataFrame(test_data, columns=None)

df.to_csv(f'gaussian_chain_{num_rows_trn}_{num_rows_tst}_{num_cols_d}_trn.csv', index=False)
df_test.to_csv(f'gaussian_chain_{num_rows_trn}_{num_rows_tst}_{num_cols_d}_test.csv', index=False)