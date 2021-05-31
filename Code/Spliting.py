
#############################################################################################################
# Program: Spliting the Dataset for Train / Val / Test folders (One time use)
# Inputs:  5 X 5 agriculture land, bareland and water patches generated from Patch.py 
# Output:  Training, Testing and validation patches splitted 
##############################################################################################################
#Spliting the Dataset
# # Creating Train / Val / Test folders (One time use)

import os
import numpy as np
import shutil
import random
root_dir = 'Dataset/' # data root path
classes_dir = ['Flood' ,'AgriLand', 'Bareland'] #total labels

val_ratio = 0.20
test_ratio = 0.10

for cls in classes_dir:
    os.makedirs(root_dir +'train/' + cls)
    os.makedirs(root_dir +'val/' + cls)
    os.makedirs(root_dir +'test/' + cls)

# Creating partitions of the data after shuffeling
src1 = root_dir + 'Flood/' # Folder to copy images from
src2 = root_dir + 'AgriLand/'
src3 = root_dir + 'Bareland/'

allFileNames = os.listdir(src1)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)* (1 - (val_ratio + test_ratio))), 
                                                           int(len(allFileNames)* (1 - test_ratio))])

allFileNames1 = os.listdir(src2)
np.random.shuffle(allFileNames1)
train_FileNames1, val_FileNames1, test_FileNames1 = np.split(np.array(allFileNames1),
                                                          [int(len(allFileNames1)* (1 - (val_ratio + test_ratio))), 
                                                           int(len(allFileNames1)* (1 - test_ratio))])

allFileNames2 = os.listdir(src3)
np.random.shuffle(allFileNames2)
train_FileNames2, val_FileNames2, test_FileNames2 = np.split(np.array(allFileNames2),
                                                          [int(len(allFileNames2)* (1 - (val_ratio + test_ratio))), 
                                                           int(len(allFileNames2)* (1 - test_ratio))])

train_FileNames = [src1+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src1+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src1+'/' + name for name in test_FileNames.tolist()]

train_FileNames1 = [src2+'/'+ name1 for name1 in train_FileNames1.tolist()]
val_FileNames1 = [src2+'/' + name1 for name1 in val_FileNames1.tolist()]
test_FileNames1 = [src2+'/' + name1 for name1 in test_FileNames1.tolist()]

train_FileNames2 = [src3+'/'+ name2 for name2 in train_FileNames2.tolist()]
val_FileNames2 = [src3+'/' + name2 for name2 in val_FileNames2.tolist()]
test_FileNames2 = [src3+'/' + name2 for name2 in test_FileNames2.tolist()]

print('Total images: ', len(allFileNames+
                            allFileNames1+allFileNames2))
print('Training: ', len(train_FileNames +
                        train_FileNames1+train_FileNames2))
print('Validation: ', len(val_FileNames + 
                          val_FileNames1+val_FileNames2))
print('Testing: ', len(test_FileNames + 
                       test_FileNames1+test_FileNames2))

# Copy-pasting images
for name1 in train_FileNames1:
    shutil.copy(name1, root_dir +'train/AgriLand')

# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, root_dir +'train/water')

# Copy-pasting images
for name2 in train_FileNames2:
    shutil.copy(name2, root_dir +'train/Bareland')

# Copy-pasting images
for name1 in val_FileNames1:
    shutil.copy(name1, root_dir +'val/AgriLand')

# Copy-pasting images
for name in val_FileNames:
    shutil.copy(name, root_dir +'val/water')

# Copy-pasting images
for name2 in val_FileNames2:
    shutil.copy(name2, root_dir +'val/Bareland')

# Copy-pasting images
for name1 in test_FileNames1:
    shutil.copy(name1, root_dir +'test/AgriLand')

# Copy-pasting images
for name in test_FileNames:
    shutil.copy(name, root_dir +'test/water')

# Copy-pasting images
for name2 in test_FileNames2:
    shutil.copy(name2, root_dir +'test/Bareland')