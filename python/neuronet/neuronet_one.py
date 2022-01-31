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

x_train_adv = np.array()

for x in x_train:
    line = np.array(x)
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
        result =




#x_train = x_train / 999
#x_test = x_test / 999

y_train_cat = keras.utils.to_categorical(y_train, 2)
y_test_cat = keras.utils.to_categorical(y_test, 2)

model = keras.Sequential([
    Dense(units=103, input_shape=(32,), activation='relu'),
    Dense(units=309, input_shape=(103,), activation='relu'),
    Dense(units=927, input_shape=(309,), activation='relu'),
    Dense(units=20, activation='softmax'),
])

print(model.summary())      # вывод структуры НС в консоль

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

model.evaluate(x_test, y_test)






