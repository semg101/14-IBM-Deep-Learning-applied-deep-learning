#Classifying European Money Denominations Loading the Data Keras
#Import the Training Data-------------------------------------------------------
'''
The following lines of Code will import the Train Data and uncompress it. 
The actual images will be stored in the directory /resources/data/train_data_keras. There will be 70 images 0.jpeg, 1.jpeg,.., 69.jpeg. 
'''
# This is for downloading training dataset. If you already have the dataset, you can comment it out in order to avoid the second time downloading.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

!wget --quiet -O /resources/data/train_data_keras.tar.gz https://cocl.us/DL0320EN_TRAIN_TAR_KERAS
!tar -xzf /resources/data/train_data_keras.tar.gz -C /resources/data --exclude '.*'

#Import the Validation Data---------------------------------------------------
'''
The following lines of Code will import the Validation Data and uncompress it. 
The actual images will be stored in the directory /resources/data/validation_data_keras. There will be 70 images 0.jpeg, 1.jpeg,.., 69.jpeg.
'''
# This is for downloading validation dataset. If you already have the dataset, you can comment it out in order to avoid the second time downloading.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

!wget --quiet -O /resources/data/validation_data_keras.tar.gz https://cocl.us/DL0320EN_VALID_TAR_KERAS
!tar -xzf /resources/data/validation_data_keras.tar.gz -C /resources/data --exclude '.*'

#Import the Test Data-----------------------------------------------------------------------
'''
The following lines of Code will import the Test Data and uncompress it. 
The actual images will be stored in the directory /resources/data/test_data_keras. There will be 70 images 0.jpeg, 1.jpeg,.., 69.jpeg. 
'''
# This is for downloading test dataset. If you already have the dataset, you can comment it out in order to avoid the second time downloading.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

!wget --quiet -O /resources/data/test_data_keras.tar.gz https://cocl.us/DL0320EN_TEST_TAR_KERAS
!tar -xzf /resources/data/test_data_keras.tar.gz -C /resources/data --exclude '.*'

'''
The folder structure for the dataset folder is train_data_keras/test_data_keras/validation_data_keras -> 5/10/20/50/100/200/500 -> FILE_NAME.jpeg

When exporting the images from folder, please use some similar relative path as follow: e.g. "/resources/data/train_data/5/0.jpeg"
'''
# The libraries you need for plotting images

import matplotlib.pyplot as plt
from PIL import Image

'''
Question 1.1
Did you know? IBM Watson Studio lets you build and deploy an AI solution, using the best of open source and IBM software and giving your team a single environment to work in. Learn more here.

Load and plot sample 0.jpeg from(/resources/data/train_data_keras/5/0.jpeg). You can use the function img = Image.open(Input) from the PIL library to load the image. The argument is one string that includes the name of the directory and the name of the image and its extension in the following form: Input = /directory_name/image_name.jpeg The function returns the value img. Plot the image you can use the plt.imshow(img). Remember the image as you will be asked to identify it later.

# Question 1.1

​

# Type your code here

Question 1.2

Load and plot 52.jpeg from (/resources/data/train_data_keras/200/52.jpeg).

# Question 1.2

​

# Type your code here

Question 1.3

Load and plot 0.jpeg from (/resources/data/validation_data_keras/5/0.jpeg).

# Question 1.3

​

# Type your code here

Question 1.4

Load and plot 36.jpeg from (/resources/data/validation_data_keras/50/36.jpeg).

# Question 1.4

​

# Type your code here
'''