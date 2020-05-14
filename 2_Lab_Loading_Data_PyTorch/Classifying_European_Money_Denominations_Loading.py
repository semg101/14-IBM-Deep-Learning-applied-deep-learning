#Classifying European Money Denominations Loading the Data PyTorch
#Import the Training Data------------------------------------------------------------------------
'''
The following lines of Code will import the Training Data and decompress it. 
The actual images will be stored in the directory /resources/data/training_data_pytorch. There will be 70 images 0.jpeg, 1.jpeg,.., 69.jpeg. 
'''
# This is for downloading training dataset. If you already have the dataset, you can comment it out in order to avoid the second time downloading.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

!wget --quiet -O /resources/data/training_data_pytorch.tar.gz https://cocl.us/DL0320EN_TRAIN_TAR_PYTORCH
!tar -xzf  /resources/data/training_data_pytorch.tar.gz -C /resources/data

#Import the Validation Data---------------------------------------------------------------------------------------
'''
The following lines of Code will import the Validation Data and decompress it. 
The actual images will be stored in the directory /resources/data/validation_data_pytorch. There will be 70 images 0.jpeg, 1.jpeg,.., 69.jpeg.
'''
# This is for downloading validation dataset. If you already have the dataset, you can comment it out in order to avoid the second time downloading.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

!wget --quiet -O /resources/data/validation_data_pytorch.tar.gz https://cocl.us/DL0320EN_VALID_TAR_PYTORCH
!tar -xzf  /resources/data/validation_data_pytorch.tar.gz -C /resources/data

#Import the Test Data---------------------------------------------------------------------------------------------
'''
The following lines of Code will import the Test Data and compress it. 
The actual images will be stored in the directory /resources/data/test_data_pytorch. There will be 70 images 0.jpeg, 1.jpeg,.., 69.jpeg.
'''
# This is for downloading test dataset. If you already have the dataset, you can comment it out in order to avoid the second time downloading.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

!wget --quiet -O /resources/data/test_data_pytorch.tar.gz https://cocl.us/DL0320EN_TEST_TAR_PYTORCH
!tar -xzf /resources/data/test_data_pytorch.tar.gz -C /resources/data --exclude '.*'

#You will need the following libraries to answer the next questions:
# Import the libraries for plotting images

import matplotlib.pyplot as plt
from PIL import Image

'''
Question 1.1
Did you know? IBM Watson Studio lets you build and deploy an AI solution, using the best of open source and IBM software and giving your team a single environment to work in. Learn more here.

Load and plot sample one 0.jpeg from the training data. You can use the function img = Image.open(Input) from the PIL library to load the image. The argument is one string that includes the name of the directory and the name of the image and its extension in the following form: Input= "/directory _name/image_name.jpeg" The function returns the value img. In order to plot the image, you can use the plt.imshow(img).

Remember the image as you will be asked to identify it in the quiz.

# Question 1.1

​

# Type your code here

Question 1.2

Load and plot sample 53 (52.jpeg) from the training data.

# Question 1.2

​

# Type your code here

Question 1.3

Load and plot sample 1 (0.jpeg) from the validation data. The directory is given in the variable validation_dir as a string.

# Question 1.3

​

# Type your code here

Question 1.4

Load and plot sample 35 (36.jpeg) from the validation data. The directory is given in the variable validation_dir as a string.

# Question 1.4

​

# Type your code here
'''
