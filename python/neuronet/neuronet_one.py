import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd
import csv

data_education_pd = pd.read_csv("C:/Users/Stran/Documents/MM_Python/python/trein/ee.csv", encoding='utf-8', delimiter=';')
data_test_pd = pd.read_csv("C:/Users/Stran/Documents/MM_Python/python/trein/tt.csv", encoding='utf-8', delimiter=';')

data_education_np = np.array(data_education_pd)
data_test_np = np.array(data_test_pd)

x_train = np.array(data_education_np[:, :32])
y_train = np.array(data_education_np[:, 32:]).astype(float)

x_test = np.array(data_test_np[:, :32])
y_test = np.array(data_test_np[:, 32:]).astype(float)


x_train_adv = np.empty([2, 960])

for x in range(len(x_train)):
    line = x_train[x]
    r = np.array([-1])
    for y in line:
        st = str(y)
        h = 0
        t = 0
        o = int(st[-1])
        if len(st) >= 2:
            t = int(st[-2])
        if len(st) >= 3:
           h = int(st[-3])
        h_cat = keras.utils.to_categorical(h, 10)
        t_cat = keras.utils.to_categorical(t, 10)
        o_cat = keras.utils.to_categorical(o, 10)
        result = np.append(h_cat, t_cat)
        result = np.append(result, o_cat)
        r = np.append(r, result)
    r = np.delete(r, 0)
    r = np.expand_dims(r, axis = 0)
    x_train_adv = np.concatenate((x_train_adv, r), axis=0)
    print(x)

x_train_adv = np.delete(x_train_adv, 0, 0)
x_train_adv = np.delete(x_train_adv, 0, 0)

x_test_adv = np.empty([2, 960])

for x in range(len(x_test)):
    line = x_test[x]
    r = np.array([-1])
    for y in line:
        st = str(y)
        h = 0
        t = 0
        o = int(st[-1])
        if len(st) >= 2:
            t = int(st[-2])
        if len(st) >= 3:
           h = int(st[-3])
        h_cat = keras.utils.to_categorical(h, 10)
        t_cat = keras.utils.to_categorical(t, 10)
        o_cat = keras.utils.to_categorical(o, 10)
        result = np.append(h_cat, t_cat)
        result = np.append(result, o_cat)
        r = np.append(r, result)
    r = np.delete(r, 0)
    r = np.expand_dims(r, axis = 0)
    x_test_adv = np.concatenate((x_test_adv, r), axis=0)
    print(x)

x_test_adv = np.delete(x_test_adv, 0, 0)
x_test_adv = np.delete(x_test_adv, 0, 0)

#x_train = x_train / 999
#x_test = x_test / 999

y_train_cat = keras.utils.to_categorical(y_train, 3)
y_test_cat = keras.utils.to_categorical(y_test, 3)

model = keras.Sequential([
    Dense(units=90, input_shape=(960,), activation='relu'),
    Dense(units=3, activation='softmax'),
])

print(model.summary())      # вывод структуры НС в консоль

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


model.fit(x_train_adv, y_train_cat, batch_size=32, epochs=50, validation_split=0.2)

model.evaluate(x_test_adv, y_test_cat)






