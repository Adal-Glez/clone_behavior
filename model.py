import csv
import numpy as np
import cv2

import sklearn
from sklearn.utils import shuffle

lines =[]

with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
   
print(len(lines))
print(len(train_samples))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(name)
                    #center_angle = float(batch_sample[3])
                    images.append(center_image)
                    correction=0
                    if i==1:
                        correction=0.2
                    if i==2:
                        correction=-0.2
                    angle = float(batch_sample[3])+correction
                    angles.append(angle)
                    #angles.append(center_angle)

            # trim image to only see section with road
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

#epochs = 3
ch, row, col = 3, 80, 320
#input_shape = (row, col, ch)
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
#model.fit(X_train, y_train, validation_split=0.2, shuffle= True, nb_epoch=epochs)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
print("save model")

model.save('model.h5')
exit()


