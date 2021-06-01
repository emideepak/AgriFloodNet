
##################################################################################
# Program: Training the CNN for Agriculture land, Bareland and water patches identification
# Inputs:  5 X 5 SAR S1 and Multispectral S2 image patches (Splitting.py code's output)
# Output:  Trained model for senitinel 1 and 2 image patches (Sen1model.h5 and Sen2model.h5)
###################################################################################
from Code import AgriFloodNet_CNN
from keras.preprocessing.image import ImageDataGenerator
#applying all the transformation we want to apply to training data set
train_datagen = ImageDataGenerator(rotation_range=30,rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#Rescling the test data set images to use for validation.
valid_datagen= ImageDataGenerator(rotation_range=30,rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
#Getting My training data ready for validation, so it will read all the data with the px size we gave.

training_set= train_datagen.flow_from_directory(directory= 'Dataset/train',
                                               target_size=(5,5), # As we choose 64*64 for our convolution model
                                               batch_size=10,
                                               class_mode='categorical' # for 2 class binary 
                                               )

#Getting My test data ready for validation, so it will read all the data with the px size we gave.

valid_set= valid_datagen.flow_from_directory(directory= 'Dataset/val',
                                               target_size=(5,5), # As we choose 64*64 for our convolution model
                                               batch_size=10,
                                               class_mode='categorical' # for 2 class binary
                                          )

#Rescling the test data set images to use for validation.
test_datagen= ImageDataGenerator(rotation_range=30,rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#Getting My test data ready for validation, so it will read all the data with the px size we gave.

test_set= test_datagen.flow_from_directory(directory= 'Dataset/test',
                                               target_size=(5,5), # As we choose 64*64 for our convolution model
                                               batch_size=10,
                                               class_mode='categorical' # for 2 class binary
                                          )

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history=model.fit(training_set, #training data to fit # Data in training set
                        epochs=100,steps_per_epoch=len(training_set), # No of epochs to run
                        validation_data=valid_set, # Test or validation set
                        validation_steps=len(valid_set)) # no of data point for validation
model.save("Sen1model.h5")
print("Saved model to disk")

test_dir = 'Dataset/test'
train_dir='Dataset/train'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(5, 5),
    batch_size=10,
    class_mode='categorical')

train_loss, train_acc = model.evaluate(train_generator, steps=len(train_dir))
print('train acc:', train_acc*100)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(5, 5),
    batch_size=10,
    class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator, steps=len(test_dir))
print('test acc:', test_acc*100)

import pandas as pd
hist_df = pd.DataFrame(history.history) 
# save to csv:  
hist_csv_file = 'Modelsen1.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# list all data in history
import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#plt.plot(100)
#plt.plot(100)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
