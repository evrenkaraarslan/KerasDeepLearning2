from __future__ import absolute_import, division, print_function,unicode_literals

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import feature_column

from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

import tensorflow

from sklearn.model_selection import cross_val_score

import numpy

import keras

from keras.layers import Input, Dense

from keras.models import Model

from keras.models import Sequential

from numpy import loadtxt

from keras.callbacks import TensorBoard

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import MinMaxScaler

import h5py

line_count=0

dataset=loadtxt(r'C:\Users\mustafa karaarslan\Downloads\data3.csv',delimiter=";",dtype=str)

dataset = np.delete(dataset, 0, 1)

print(type(dataset))

map(int,dataset)

print(dataset.dtype)

evren = dataset.astype(np.float)

xinput = evren[1:,0:2]

youtput = evren[1:,2]

youtput=np.reshape(youtput, (-1,1))

scaler_xinput = MinMaxScaler()

scaler_youtput = MinMaxScaler()

print(scaler_xinput.fit(xinput))

xinputscale=scaler_xinput.transform(xinput)

print(scaler_youtput.fit(youtput))

youtputscale=scaler_youtput.transform(youtput)

xinput_train, xinput_test, youtput_train, youtput_test =train_test_split(xinputscale, youtputscale)
for row in dataset:

    if line_count == 0:

        print(f'Column names are {", ".join(row)}')

        line_count += 1

    else:

        print(f'\t Current = {row[0]} , Voltage = {row[1]},Output Type={row[2]}')

        line_count += 1

print(line_count)

keras.callbacks.callbacks.LambdaCallback(on_epoch_begin=None,
                                         on_epoch_end=None,
                                         on_batch_begin=None,
                                         on_batch_end=None,
                                         on_train_begin=None,
                                         on_train_end=None)

class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self,logs={}):

         self.losses = []

    def on_batch_end(self,batch,logs={}):

         self.losses.append(logs.get('loss'))

model = Sequential()

model.add(Dense(12, input_dim=(2), activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

history=LossHistory()

A=model.fit(xinput_train, youtput_train , epochs=80, batch_size=32,verbose=1,callbacks=[history])

model.save("EvrenTrainedData.h5")

dataset1=np.loadtxt(r'C:\Users\mustafa karaarslan\Pictures\data7.csv',delimiter=";",dtype=str)

dataset1 = np.delete(dataset1, 0, 1)

print(type(dataset1))

map(int,dataset1)

print(dataset1.dtype)

evren1 = dataset1.astype(np.float)

print("evren data type" ,evren1.dtype)

xnewinput = evren1[1:,0:2]

xnewinput= scaler_xinput.transform(xnewinput)

ynew= model.predict(xnewinput)

xnewinput = scaler_xinput.inverse_transform(xnewinput)

for i in range(len(xnewinput)):

    print("Current,Voltage=%s, Gas Pressure is %s" % (xnewinput[i], ynew[i]))

    if ynew[i]>0.5:

        print("Gas Pressure is not enough!")

    else:

        print("Gas Pressure is sufficient")

model.summary()

print("THIS IS HISTORY LOSSES:",history.losses)

C = np.arange(0, 80)

plt.figure()

plt.plot(C, A.history["loss"], label="LOSS",color="lime")

plt.plot(C, A.history["accuracy"], label="ACCURACY",color="yellow")

plt.xlabel("Number of Epoch")

plt.ylabel("Loss and Accuracy")

plt.legend()

plt.savefig("evren")

plt.show()