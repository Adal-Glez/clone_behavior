import csv
import numpy as np
import cv2

lines =[]
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images =[]
measurements  = []
for line in lines:
    for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = "data/IMG/"+filename
            image = cv2.imread(current_path)
            images.append(image)
            correction=0
            if i==1:
                correction=0.2
            if i==2:
                correction=-0.2
            measurement = float(line[3])+correction
            measurements.append(measurement)

augmented_images, augmented_mesurements = [],[]

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_mesurements.append(measurement)
    augmented_images.append(np.fliplr(image)) # cv2.flip(image,1)
    augmented_mesurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_mesurements)

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D

epochs = 5


#based on NVIDIA Architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping((70,25),(0,0))))#70rowTop,25B,L,R
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

       
model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle= True, nb_epoch=epochs)

model.save('model.h5')
exit()





















