import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd
import csv

data_education_pd = pd.read_csv(os.path.abspath("train/education.csv"), encoding='utf-8', delimiter=';')
data_test_pd = pd.read_csv(os.path.abspath("train/test.csv"), encoding='utf-8', delimiter=';')

data_education_np = np.array(data_education_pd)
data_test_np = np.array(data_test_pd)

x_train = np.array(data_education_np[:, :32])
y_train = np.array(data_education_np[:, 32:])

x_test = np.array(data_test_np[:, :32])
y_test = np.array(data_test_np[:, 32:])

x_train = x_train / 999
x_test = x_test / 999

y_train_cat = keras.utils.to_categorical(y_train, 20)
y_test_cat = keras.utils.to_categorical(y_test, 20)

model = keras.Sequential([
    Dense(units=128, input_shape=(32,), activation='relu'),
    Dense(units=1280, input_shape=(128,), activation='relu'),
    Dense(20, activation='softmax')
])

print(model.summary())      # вывод структуры НС в консоль

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

history= model.fit(x_train, y_train_cat, batch_size=32, epochs=400, validation_split=0.2)

model.evaluate(x_test, y_test_cat)

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
count = 0

for x in range(len(y_test)):
    if (pred[x]-y_test[x]==0 or pred[x]-y_test[x]==1 or pred[x]-y_test[x]==-1):
        count=count+1
        raz =pred[x]-y_test[x]
        print(str(count)+'.'+str(pred[x])+ ' - '+str(y_test[x])+ ' = ' + str(raz))
