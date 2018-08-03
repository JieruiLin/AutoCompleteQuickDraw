from keras.models import model_from_json
from scipy.ndimage import imread
import numpy as np


model = None
with open('model.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('weights.h5')

rgba = imread('test2.png', mode='RGBA')
img_array = np.zeros((100, 100), dtype=np.int16)
for x in range(100):
    for y in range(100):
        img_array[x][y] = rgba[x][y][3]

f = open('kek.txt', 'w')
for i in range(100):
    for j in range(100):
        if img_array[i][j] > 0:
            f.write('X')
        else:
            f.write('_')
    f.write('\n')
f.close()

img_array = img_array.flatten()
prediction_confidence = model.predict(img_array.reshape(1, 100, 100, 1))
prediction = np.argmax(prediction_confidence)
print("confidence: " + str(prediction_confidence))
print("prediction: " + str(prediction))
