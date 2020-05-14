#Classifying European Money Denominations: Training a Pre-trained model 
#Preparation--------------------------------------------------------------------------
# You can comment out this box when you already have the dataset
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

# Download Training Dataset
!wget --quiet -O /resources/data/train_data_keras.tar.gz https://cocl.us/DL0320EN_TRAIN_TAR_KERAS
!tar -xzf /resources/data/train_data_keras.tar.gz -C /resources/data --exclude '.*'

# Download Validation Dataset
!wget --quiet -O /resources/data/validation_data_keras.tar.gz https://cocl.us/DL0320EN_VALID_TAR_KERAS
!tar -xzf /resources/data/validation_data_keras.tar.gz -C /resources/data --exclude '.*'

# Keras Modules
import keras
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Model

# Non-Keras Modules
import os
from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

#Create Image Dataset Generator--------------------------------------------------------------------
# Store image dataset in the directory
train_data_dir = '/resources/data/train_data_keras'
validation_data_dir = '/resources/data/validation_data_keras'
classes = ['5', '10', '20', '50', '100', '200', '500']

#Using ImageDataGenerator().flow_from_directory() to load the image from directory and generate the training dataset.
# Create the ImageDataGenerator for training dataset

train_generator = ImageDataGenerator().flow_from_directory(train_data_dir
                                                           , target_size=(224, 224)
                                                           , batch_size=10
                                                           , classes=classes
                                                           , seed=0
                                                           , shuffle=True)

#Using ImageDataGenerator().flow_from_directory() to load the image from directory and generate the validation dataset.
# Create the ImageDataGenerator for validation dataset

valid_generator = ImageDataGenerator().flow_from_directory(validation_data_dir
                                                           , target_size=(224, 224)
                                                           , batch_size=5
                                                           , classes=classes
                                                           , seed=0
                                                           , shuffle=True)

#Questions---------------------------------------------------------------
'''
Question 3.1: Preparation
5 point

Step 1: Load the pre-trained model ResNet50. Set weights='imagenet'

# Step 1: Load the pre-trained model ResNet50

​

# Type your code here

Step 2: The following lines of code sets the attribute trainable to False. As a result, the parameters in these layers will not be affected by training.

# Step2: Set parameters in pre-train model to False

​

# Type your code here

Step 3: ResNet50 is used to classify 1000 different objects; as a result, the last layer has 1000 outputs. However, you are going to classify 7 different classes, so you need to remove the last layer and replace with the new classification layer.

# Step 3: Replace the old classification layer with the new classification layer

​

# Type your code here

Print out the model in order to show whether you get the correct answer.
(Your peer reviewer is going to mark based on what you print here.)

# Print the model (PLEASE DO NOT MODIFY THIS BOX)

​

model.summary()

Question 3.2: Train the model
5 points
Did you know? IBM Watson Studio lets you build and deploy an AI solution, using the best of open source and IBM software and giving your team a single environment to work in. Learn more here.

Step 1: Use the model.compile() to set the configuration for the model. Use the optimizer Adam, loss function categorical_crossentropy and metrics = ['accuracy'] as the parameters for training the model.

# Step 1: Use model.compile() to set the configuration

​

# Type your code here

Step 2: Train the model with 20 epochs.

# Step 2: Train the model

​

# Type your code here

The information of loss and accuarcy for both training and validation is in model.history.history. Get the training history, and store the history into train_history.

# Get the training history

​

train_history = model.history.history

Step 3: Plot out the result of loss for both training and validation.
(Your peer reviewer is going to mark based on what you plot here.)

# Step 3: Plot the loss for both training and validation

​

# Type your code here

Step 4: Plot out the result of accuracy for both training and validation.
(Your peer reviewer is going to mark based on what you plot here.)

# Step 4: Plot the accuracy for both training and validation

​

# Type your code here

Question 3.3: Plot 5 Random Images with their predictions
3 points

Use the validation_dataset to see whether model can predict. (You will use the test_dataset in the future lab. Use the validation_dataset for now.) Notice that you need to set the shuffle to False, so the the order of predictions will be the same as the file loading order.

# Generate test dataset and generate the prediction results

​

test_valid_generator = ImageDataGenerator().flow_from_directory(validation_data_dir

                                                           , target_size=(224, 224)

                                                           , batch_size=5

                                                           , classes=classes

                                                           , seed=0

                                                           , shuffle=False)

Create a test dataset using validation data. Plot 5 random images which index is in the numbers list. Also print the predicted label and print a string indicate whether it has been correctly classified or mis-classified.
(Your peer reviewer is going to mark based on what you plot here.)

Steps:

    Use MODEL.predict_generator() to do the prediction
    Use np.argmax() to find the location of the maximum value
    Get the class labels from train_generator using train_generator.filenames
    Get the class indices from the previous step using train_generator.class_indices
    Match classes with labels like [0, 1, 2, 3, 4, 5, 6] -> ['5', '10', '20', '50', '100', '200', '500']
    Get the true label by analysis folder name on the image resource path. Notice the test_valid_genertor.filename order is the same as the predicted order as you did not shuffle the dataset.
    Compare, plot and print the result

# Plot five random images and their predictions

​

random.seed(0)

numbers = [random.randint(0, 69) for i in range(0, 5)]

​

# Type your code here

Question 3.4: Use the second model VGG16 to do the prediction
5 points

Repeat the steps in Question 3.1, 3.2 to predict the result using VGG16 model. Then, print out the last validation accuracy.

Steps:

    Load the pre-trained model VGG16
    Replace the last classification layer with only 7 classes
    Set the configuration (parameters) using MODEL.compile()
    Train the model
    Print the last validation accuracy

Hint:

    Use optimizer Adam
    Use loss function categorical_crossentropy
    Epoches = 5

You are welcome to try any pattern of setting and find out the best result. Please name the model variable as model_vgg.
(Your peer reviewer is going to mark based on what you print here.)

# Use VGG16 to train the model and print out the last validation accuracy.

​

# Type your code here

Save the trained model

Save the trained model for the following chapters

# Save the model

​

model.save("resnet50_keras.pt")

model_vgg.save("vgg16_keras.pt")
'''