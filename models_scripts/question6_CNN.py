#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:30:59 2020

@author: natewagner
"""


import pandas as pd
import numpy as np



all_train = pd.read_csv("/Users/natewagner/Documents/Surveys/question6_data_w_actual.csv")
all_train_data = all_train[all_train.columns[:2500]]
act = all_train[all_train.columns[2500]]
all_train_data = question6_data


images = []
for index, row in all_train_data.iterrows():
    images.append(np.array(row[:2500]).reshape(50, 50))

#for x in range(1, 21):
#    print(Image.fromarray(images[0].astype(np.uint8)).show())


from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(
 images, act, test_size=0.20, random_state=0)



train_img = np.stack(train_img, axis=0)
test_img = np.stack(test_img, axis=0)


#reshape data to fit model
train_img = train_img.reshape(2400,50,50,1)
test_img = test_img.reshape(600,50,50,1)



train_img =  train_img/255
test_img = test_img/255



from keras.utils import to_categorical
#one-hot encode target column
train_lbl = to_categorical(train_lbl)
test_lbl = to_categorical(test_lbl)
train_lbl[0]


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
#create model
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(50,50,1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))




#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_img, train_lbl, validation_data=(test_img, test_lbl), epochs=1)


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()




# evaluate model
score = model.evaluate(test_img, test_lbl, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])






# confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(test_img)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
print(Y_pred_classes)
Y_true = np.argmax(test_lbl,axis = 1)
print(Y_true)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()



# to save model
model.save('/Users/natewagner/Documents/Surveys/Models/Question6_CNN.h5')


from tensorflow import keras

# to reload model
Question6_CNN = keras.models.load_model('/Users/natewagner/Documents/Surveys/Models/Question6_CNN.h5')





