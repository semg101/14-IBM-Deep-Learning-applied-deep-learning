#Classifying European Money Denominations
#Preparation------------------------------------------------------------
#Download the datasets you needed for this lab.
# You can comment the code in this box out if you already have the dataset.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

# Download Training Dataset
!wget --quiet -O /resources/data/train_data_keras.tar.gz https://cocl.us/DL0320EN_TRAIN_TAR_KERAS
!tar -xzf /resources/data/train_data_keras.tar.gz -C /resources/data --exclude '.*'

# Download Validation Dataset
!wget --quiet -O /resources/data/validation_data_keras.tar.gz https://cocl.us/DL0320EN_VALID_TAR_KERAS
!tar -xzf /resources/data/validation_data_keras.tar.gz -C /resources/data --exclude '.*'

# Import Keras Modules

import keras
from keras.preprocessing.image import ImageDataGenerator

# Import Non-Keras Modules

import os
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image
import numpy as np 

# Parameters for the image dataset generators

TARGET_SIZE = (224, 224)
BATCH_SIZE = 5
CLASSES = ['5', '10', '20', '50', '100', '200', '500']
RANDOM_SEED = 0

#Load Image-------------------------------------------------------------------------------
#Training Images
#The train images are stored in the following directory /resources/data/train_data_keras/. We can save it in the variable train_data_dir.
# Get the train dataset folder and store it in train_data_dir

train_data_dir = '/resources/data/train_data_keras'

'''
To load the image, you need the directory that contains the entire dataset. Then you can use the ImageDataGenerator() with flow_from_directory() to get the images from folder and load them into a data object.
In function flow_from_directory(), you can set the directory path that contains the entire dataset.
You can set the target size of images, set the batch size, set the classes which are the labels in the dataset and the random seed.
The labels are the sub-folder names like 5, 10, 20 ... In our dataset, there are 7 different labels. (5, 10, 20, 50, 100, 200, 500)
'''
# Generate training image dataset

train_generator = ImageDataGenerator().flow_from_directory(train_data_dir
                                                           , target_size=TARGET_SIZE
                                                           , batch_size=BATCH_SIZE
                                                           , classes=CLASSES
                                                           , seed=RANDOM_SEED)

#Validation Images
#The validation data is stored in the following directory /resources/data/validation_data_keras. We can assign it to the variable validation_data_dir.
# Get the validation dataset folder and store it in validation_data_dir

validation_data_dir = '/resources/data/validation_data_keras'

'''
Questions

In this section, you will test your dataset object. Please plot out the images in the first batch.

If you generate the dataset correctly, there are 5 European Bills are going to be plot out.

Hint:

    Use image_generator.batch_size to get the batch size.
    Create a loop to get each image
    Use image_generator.next()[0] to get the first batch
    Convert the image array to np.uint8 by obj.astype(np.uint8) before you plot them
    Use the plt.imshow(image) and plt.show() to plot the image

Question 2.1

Test the training dataset.

# Question 2.1

​

# Type your code here

Question 2.2

Test the validation dataset.

# Question 2.2

​

# Type your code here
'''