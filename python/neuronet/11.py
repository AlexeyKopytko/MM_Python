import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

c = np.empty([3, 2])
f = np.array([[-40, 14]])

#d = np.concatenate((c, f), axis=0)
print(c)