import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, LayerNormalization,Dropout
from keras.callbacks import TensorBoard  

from keras import layers
import matplotlib.pyplot as plt
import numpy as np
class1 = np.load("mixue.npy")
class2=np.load("handsand1legup.npy")
print(class1.shape)
print(class2.shape)

x1=[]
x2=[]

nsample1=len(class1)
nsample2=len(class2)

no_of_timesteps=40
for i in range(no_of_timesteps, nsample1):
    x1.append(class1[i-no_of_timesteps:i,:])

for i in range(no_of_timesteps, nsample2):
    x2.append(class2[i-no_of_timesteps:i,:])

print(np.array(x1).shape)
print(np.array(x2).shape)

Final=np.concatenate((x1,x2))
#Final1 = pd.concat([np.array(class1), np.array(FinalFHGSData)])
print(Final.shape)
Class1_Label = np.zeros((84), dtype="int").reshape(-1,1)
Class2_Label = np.ones((46), dtype="int").reshape(-1,1)
#NHGS_label=np.full(1,0).reshape(-1,1)
print(Class1_Label.shape)
labels = np.vstack([Class1_Label, Class2_Label])

ohe = OneHotEncoder()
y= ohe.fit_transform(labels).toarray().astype(int)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(Final,y, test_size = 0.20, shuffle= True)

log_dir = os.path.join("Logs")

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(40,51), recurrent_dropout=0.2))
model.add(LayerNormalization(axis=1))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LayerNormalization(axis=1))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, validation_data = (X_test, y_test),batch_size=16, epochs=100)

model.save('mmodelfinal1.h5')  # creates a HDF5 file 'my_model.h5'