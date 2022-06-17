train_data_points = []
test_data_points = []

from random import random
import pandas as pd
import numpy as np
from math import sin,cos

k = 5

for i in range(4500):
    row = [0]
    for _ in range(k):
        theta = np.random.uniform(0,2*np.pi)
        x = 0.5 + sin(theta)
        y = 0.5 + cos(theta)
        row.append(x)
        row.append(y)
    train_data_points.append(row)
    row = [1]
    for _ in range(k):
        theta = np.random.uniform(0,2*np.pi)
        x = -1*0.5 + 2*sin(theta)
        y = -1*0.5 + 2*cos(theta)
        row.append(x)
        row.append(y)
    train_data_points.append(row)

for i in range(500):
    row = [0]
    for _ in range(k):
        theta = np.random.uniform(0,2*np.pi)
        x = 0.5 + sin(theta)
        # x = sin(theta)
        y = 0.5 + cos(theta)
        # y = cos(theta)
        row.append(x)
        row.append(y)
    test_data_points.append(row)
    row = [1]
    for _ in range(k):
        theta = np.random.uniform(0,2*np.pi)
        x = -1*0.5 + 2*sin(theta)
        # x =  2*sin(theta)
        y = -1*0.5 + 2*cos(theta)
        # y =  2*cos(theta)
        row.append(x)
        row.append(y)
    test_data_points.append(row)

df = pd.DataFrame(train_data_points,columns=None)
df.to_csv('2d_train.csv',index=False)
df = pd.DataFrame(test_data_points,columns=None)
df.to_csv('2d_test.csv',index=False)
