#Classifying European Money Denominations: Comparing Two Models
#Preparation----------------------------------------------------------
#Download the datasets you needed for this lab.
# You can comment out this box when you already have the dataset
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

# Download test dataset
!wget --quiet -O /resources/data/test_data_keras.tar.gz https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0320EN/Datasets/Keras/test_data_keras.tar.gz
!tar -xzf /resources/data/test_data_keras.tar.gz -C /resources/data --exclude '.*'

# Import Keras Modules
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# Import Non-Keras Modules
import os
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image
import numpy as np 

#Create Dataset Class and Object--------------------------------------------------------
# The path for accessing image dataset folder
test_data_dir = '/resources/data/test_data_keras/'

#Try-----------------------------------------------
#Try to construct a test_generator using test_data_dir as directory path
# Create Dataset Class
CLASSES = ['5', '10', '20', '50', '100', '200', '500']

test_generator = ImageDataGenerator().flow_from_directory(test_data_dir
                                                          , target_size=(224, 224)
                                                          , batch_size=5
                                                          , classes=CLASSES
                                                          , seed=0)

#Load Pre-trained Model---------------------------------------------------------------------
#Load the ResNet50 and VGG16 model you created from the last section
# Load pre-trained model
model = load_model("resnet50_keras.pt")
model_vgg = load_model("vgg16_keras.pt")

# Print model structure
print("=================================== ResNet50 =========================================")
print("ResNet5O:\n", model.summary())
print("=================================== VGG16 ============================================")
print("VGG16:\n", model_vgg.summary())

#Analyze Models---------------------------------------------------------------------------------------
#Try---------------------------------------------
# Predict the data using ResNet50 model and print out accuracy
score = model.evaluate_generator(test_generator, test_generator.n//test_generator.batch_size)
print("Accuracy using ResNet50: ", score[1]) 

#try---------------------------------------------------------
# Predict the data using VGG16 model and print out accuracy
score = model_vgg.evaluate_generator(test_generator, test_generator.n//test_generator.batch_size)
print("Accuracy using VGG16: ", score[1]) 