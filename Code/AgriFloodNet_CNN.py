
##################################################################################
# Program: CNN model for Agriculture land, Bareland and water patches identification
###################################################################################

# import the necessary packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras import backend as K

model= Sequential()
model.add(Conv2D(8, kernel_size= (5,5), padding="same", activation='relu', input_shape=(5,5,3)))
#Second layer
model.add(Conv2D(16, kernel_size=(3,3), padding="same", activation='relu', strides=1))
#Third layer
model.add(Conv2D(32,kernel_size=(3, 3), padding="same",activation='relu', strides=1))
#Fourth layer
model.add(Conv2D(64, kernel_size=(3,3), padding="same",activation='relu', strides=1))
model.add(Dropout(0.6))
#Fifth layer
model.add(Conv2D(128, kernel_size=(3,3), padding="same",activation='relu', strides=1))
#Sixth layer
model.add(Conv2D(256, kernel_size=(3,3), padding="same",activation='relu', strides=1))
model.add(Dropout(0.8))

    
#Final layer
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu')) 
#Output layer 
model.add(Dense(3))
model.add(Activation('softmax'))
model.summary()

