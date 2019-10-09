import numpy as np
from keras.utils import to_categorical

from sklearn.metrics import classification_report

import tensorflow as tf
import keras.backend as K

from loader import Loader
from LeNet5 import lenet5

epochs = 20
batch_size = 128
num_classes = 10

loader = Loader('./data/')

x_train, y_train = loader.load_train()
y_train = to_categorical(y_train, num_classes)
np.divide(x_train, 255)
print("training data's shape is {}".format(x_train.shape))

x_test, y_test = loader.load_test()
y_test = to_categorical(y_test, num_classes)
np.divide(x_test, 255)
print("testing data's shape is {}".format(x_test.shape))

model = lenet5((32, 32, 1,), num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

history = model.predict(x_test, verbose=1)
mask = np.argmax(history, axis=1)

y_pred = np.zeros(history.shape)
for i in range(y_pred.shape[0]):
    y_pred[i][mask[i]] = 1

print(y_pred[100])
print(y_test[100])

print(classification_report(y_test, y_pred))