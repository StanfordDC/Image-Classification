%tensorflow_version 1.x
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.applications import MobileNetV2
from keras.datasets import cifar10
from tensorflow.keras import layers, callbacks

import cv2

import sys
import numpy as np
import csv
import math
import numpy as np

import matplotlib.pyplot as plt

#Import dataset
# Class names for different classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

# Load training data, labels; and testing data and their true labels
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print ('Training data size:', train_images.shape, 'Test data size', test_images.shape)

# Normalize pixel values between -1 and 1
train_images = train_images / 127.5 - 1 
test_images = test_images / 127.5 - 1 

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
img_x, img_y = 32, 32

train_images = train_images.reshape(train_images.shape[0], img_x, img_y, 3)
test_images  = test_images.reshape(test_images.shape[0], img_x, img_y, 3)
input_shape = (img_x, img_y, 3)

#Visualize Dataset
%matplotlib inline
#Show first 25 training images below
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
    
#Resize Images for use with MobileNetV2
# Upsize all training and testing images to 96x96 for use with mobile net
minSize = 96 #minimum size requried for mobileNetV2
# You may use cv2 package. Look for function:
#"cv2.resize(<originalImage>, dsize=(minSize, minSize), interpolation=cv2.INTER_AREA)"
# resize train image: You can first initialize a numpy array resized_train_images to store all the resized training images
resized_train_images = np.zeros((50000, minSize, minSize, 3), dtype=np.float32)
# <Write code for resizing>
for image in resized_train_images:
    image = cv2.resize(image, (minSize, minSize), interpolation=cv2.INTER_AREA)
# resize test image: You can first initialize a numpy array resized_test_images to store all the resized test images
resized_test_images = np.zeros((10000, minSize, minSize, 3), dtype=np.float32)
# <Write code for resizing>
for image in resized_test_images:
    image = cv2.resize(image, (minSize, minSize), interpolation=cv2.INTER_AREA)

#Download MobileNetV2 network
!pip install 'h5py<3.0.0'
base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')
                                               
#Add custom layers 
model = tf.keras.Sequential()
model.add(base_model)
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(500, activation='relu'))
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

for layer in base_model.layers:
  layer.trainable = False

#Add loss function, compile and train model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
epochs = 5
batch_size = 256
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs)

#Check and test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
