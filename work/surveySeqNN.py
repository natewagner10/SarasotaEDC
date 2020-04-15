#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:08:18 2020

@author: natewagner
"""


import pandas as pd
import numpy as np


actual_yes_no = pd.read_csv("/Users/natewagner/Documents/Surveys/train_data_with_actual.csv")
all_train_data = pd.read_csv("/Users/natewagner/Documents/Surveys/checkYN_train_current.csv")


images = []
for index, row in all_train_data.iterrows():
    images.append(np.array(row[:36100]).reshape(190,190))

    

act = actual_yes_no[actual_yes_no.columns[:1]]
#pixs = pixs / 255


np.where(np.isnan(act))
act.xs(705)[0] = 1



from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(
 images, act, test_size=0.20, random_state=0)



train_img = np.stack(train_img, axis=0)
test_img = np.stack(test_img, axis=0)


#reshape data to fit model
train_img = train_img.reshape(1200,190,190,1)
test_img = test_img.reshape(300,190,190,1)

train_img /= 255
test_img /= 255



from keras.utils import to_categorical
#one-hot encode target column
train_lbl = to_categorical(train_lbl)
test_lbl = to_categorical(test_lbl)
train_lbl = train_lbl[:,1:]
test_lbl = test_lbl[:,1:]
train_lbl[0]


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(190,190,1)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
#model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))




#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#train the model
model.fit(train_img, train_lbl, validation_data=(test_img, test_lbl), epochs=49)


score = model.evaluate(test_img, test_lbl, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])






model.predict(test_img[:4])
test_lbl[:4]

















