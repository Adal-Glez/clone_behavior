# Adalberto Gonzalez
# Date: April, 2017
# behavioral cloning model: based on data collected from driving in a simulator , this model trains a deep neural netowrk to do the driving for me (us) in a simulated enviroment.


import csv
import numpy as np
import cv2
import os

import sklearn
from sklearn.utils import shuffle

lines =[]

path = 'data_folders/'

os.chdir(path)
#each folder corresponds to a dataset that covers a specific part of the training 
PATHS = ['data/driving_log.csv',
        'center_lane/driving_log.csv',
        'center_lane_counter/driving_log.csv',
        'recovery_side/driving_log.csv',
        'smooth_curves/driving_log.csv']
# 
print(PATHS)

# the first line is the column names.
for PATH in PATHS:
    with open(PATH) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        
print("samples: ", len(lines))  

from sklearn.model_selection import train_test_split
#adjsut samples quanity 
lines = lines[:(len(lines) // 32) * 32]

print("adjusted samples: ", len(lines)) 
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print("train samples: ", len(train_samples))
print("validation samples: ", len(validation_samples))

def generator(samples, batch_size=128):
    num_samples = len(samples)
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = 'IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    
                    images.append(image)
                    #correct image from side cameras
                    if i==0:
                        angle = float(batch_sample[3])  #center
                    if i==1:
                        angle = float(batch_sample[3])+correction  #left
                    if i==2:
                        angle = float(batch_sample[3])-correction  #right
                    angles.append(angle)

                    #flipped image
                    images.append(np.fliplr(image)) 
                    angles.append(angle*-1.0)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
import matplotlib.pyplot as plt

epochs = 4

input_shape=(160,320,3)
print("setting model")
#based on NVIDIA Architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))  #input_shape=(160,320,3)
model.add(Cropping2D(cropping=((70,25),(0,0))))#70rowTop,25B,L,R
model.add(Convolution2D(24,5,5,subsample=(2,2) ,activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2) ,activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2) ,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print("compiling model")

model.compile(loss='mse', optimizer = 'adam')
print("fitting model")
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=epochs)

print("save model")
model.save('model.h5')
print("model saved")
model.summary()
exit()
