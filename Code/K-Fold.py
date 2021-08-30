from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np

# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 5, 5, 3
loss_function = categorical_crossentropy
no_classes = 3
no_epochs = 25
optimizer = Adam()
verbosity = 1
num_folds = 10

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("content/Dataset")))
random.seed(42)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = img_to_array(image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    if label=='Bareland':
       labels.append(0) 
    elif label=='NoFlood':
       labels.append(1)
    elif label=='Flood':
       labels.append(2) 
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.3, random_state=42)
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=3)
testY = to_categorical(testY, num_classes=3)

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((trainX, testX), axis=0)
targets = np.concatenate((trainY, testY), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  # Define the model architecture
  model= Sequential()
  model.add(Conv2D(8, kernel_size= (5,5), padding="same", activation='relu', input_shape=(5,5,3)))
  #Second layer
  model.add(Conv2D(16, kernel_size=(3,3), padding="same", activation='softmax', strides=1))
  #Third layer
  model.add(Conv2D(32,kernel_size=(3, 3), padding="same",activation='relu', strides=1))
  #Fourth layer
  model.add(Conv2D(64, kernel_size=(3,3), padding="same",activation='softmax', strides=1))
  model.add(Dropout(0.6))
  #Fifth layer
  model.add(Conv2D(128, kernel_size=(3,3), padding="same",activation='relu', strides=1))
  #Sixth layer
  model.add(Conv2D(256, kernel_size=(3,3), padding="same",activation='softmax', strides=1))
  model.add(Dropout(0.8))
  #Final layer
  model.add(Flatten())
  model.add(Dense(1024))
  model.add(Activation('relu')) 
  #Output layer 
  model.add(Dense(3))
  model.add(Activation('softmax'))

  # Compile the model
  model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])


  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(trainX, trainY,
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity)

  # Generate generalization metrics
  scores = model.evaluate(testX, testY, verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')