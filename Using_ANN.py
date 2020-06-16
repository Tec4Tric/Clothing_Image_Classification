'''
This code is written by Sayan De, from Tec4Tric.
This code is an example to Classify the Clothing Images from **Fashion_MNIST** dataset using Artificial Neural Network.
Google Colab link - https://bit.ly/fashion_mnist-ann 
Website - https://tec4tric.com   
Facebook - https://www.facebook.com/tec4tric  
YouTube - https://www.youtube.com/tec4tric   
Watch this tutorial - https://youtu.be/oHAkK_9UCQ8 
'''


# Importing Packages
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense
import matplotlib.pyplot as plt
%matplotlib inline

# Loading Dataset
from keras.datasets import fashion_mnist
(x_train, x_lab),(y_test, y_lab) = fashion_mnist.load_data()
plt.imshow(x_train[0])
plt.title('Class: {}'.format(x_lab[0]))
plt.figure()

# Normalizing the Data
x_train = keras.utils.normalize(x_train, axis = 1)
y_test = keras.utils.normalize(y_test, axis = 1)
plt.imshow(x_train[0])
plt.title('Class: {}'.format(x_lab[0]))
plt.figure()

# Defining the Model
model = Sequential()
model.add(Flatten(input_shape=((28,28))))
model.add(Dense(200, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compiling the Model
model.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])

# Fitting the Model
model.fit(x_train, x_lab, epochs = 20)

# Evaluating on the Test Data
model.evaluate(y_test, y_lab)

# Predict the first 10 images, Probability Distribution
p = model.predict(y_test[:10])
print(p)

pred = np.argmax(p, axis=1)
print(pred)
print(y_lab[:10])

# Visualizing the result
for i in range(10):
  plt.imshow(y_test[i], cmap="binary")
  plt.title('Original: {}, Predicted: {}'.format(y_lab[i], pred[i]))
  plt.axis("Off")
  plt.figure()
