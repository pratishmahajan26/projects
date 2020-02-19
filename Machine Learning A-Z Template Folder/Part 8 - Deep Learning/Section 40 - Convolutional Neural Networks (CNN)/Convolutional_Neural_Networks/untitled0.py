# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:09:12 2019

@author: spriyadarshini
"""

#CNN - convolutional neural network

# import the libs

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# initialising the CNN
classifier = Sequential()
#convolutional layer
classifier.add(Convolution2D(32,3,3,input_shape=(64, 64, 3), activation = 'relu' ))
# max pooling layer
classifier.add(MaxPool2D(pool_size = (2,2)))
# Flatten
classifier.add(Flatten())

# Full connection
# First hidden layer
classifier.add(Dense(output_dim =128 ,activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
# compiling all the layers
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting CNN to images
#image Augmentation - to increase the number of images - it rotates,sheer ect on the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_generator,
        validation_steps=2000)


