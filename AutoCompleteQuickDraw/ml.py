import numpy as np
import keras, os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop

train_data = []
train_files = os.listdir("train/arr")
for file_name in train_files:
    train_data.append(np.loadtxt("train/arr/" + file_name))

train_data = np.array(train_data)
train_expected_outputs = np.loadtxt('train/outputs.txt')

x_train = train_data.reshape(len(train_files), 50, 50, 1)
x_train = x_train.astype('float32')
print(str(x_train.shape))

y_train = keras.utils.to_categorical(train_expected_outputs, 10)
print(str(y_train.shape))

test_data = []
test_files = os.listdir("test/arr")
for file_name in test_files:
    test_data.append(np.loadtxt("test/arr/" + file_name))

test_data = np.array(test_data)
test_expected_outputs = np.loadtxt('test/outputs.txt')

x_test = test_data.reshape(len(test_files), 50, 50, 1)
y_test = keras.utils.to_categorical(test_expected_outputs, 10)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(2500, activation='relu', input_shape=(50, 50, 1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=2, epochs=15, verbose=1)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
