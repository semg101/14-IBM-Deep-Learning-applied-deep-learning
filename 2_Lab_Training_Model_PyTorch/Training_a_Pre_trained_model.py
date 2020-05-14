#Classifying European Money Denominations: Training a Pre-trained model
#Preparation------------------------------------------------------------------------------------
# You can comment out this box when you already have the dataset
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

# Download Training Dataset
!wget --quiet -O /resources/data/training_data_pytorch.tar.gz https://cocl.us/DL0320EN_TRAIN_TAR_PYTORCH
!tar -xzf  /resources/data/training_data_pytorch.tar.gz -C /resources/data --exclude '.*'

# Download Validation Dataset
!wget --quiet -O /resources/data/validation_data_pytorch.tar.gz https://cocl.us/DL0320EN_VALID_TAR_PYTORCH
!tar -xzf  /resources/data/validation_data_pytorch.tar.gz -C /resources/data --exclude '.*'

# Import PyTorch Modules will be used in the lab
import torch 
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas
from torchvision import transforms
import torch.nn as nn
torch.manual_seed(0)

# Import Non-PyTorch Modules will be used in the lab
import time
from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

#Create Dataset Class and Object-------------------------------------------------------------------------------------------
# Url that contains CSV files
train_csv_file = 'https://cocl.us/DL0320EN_TRAIN_CSV'
validation_csv_file = 'https://cocl.us/DL0320EN_VALID_CSV'

# Absolute path for finding the directory contains image datasets
train_data_dir = '/resources/data/training_data_pytorch/'
validation_data_dir = '/resources/data/validation_data_pytorch/'

# Create Dateaset Class

class Dataset(Dataset):
    
    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_name = pd.read_csv(csv_file)
        self.len = self.data_name.shape[0] 
    
    # Get Length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        img_name = self.data_dir + self.data_name.iloc[idx, 2]
        image = Image.open(img_name)
        y = self.data_name.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return image, y

# Construct the composed object for transforming the image
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224))
                               , transforms.ToTensor()
                               , transforms.Normalize(mean, std)])

'''
Create a training dataset and validation dataset object using the csv file stored in the variables the train_csv_file and validation_csv_file. 
The directories are stored in the variable train_data_dir and validation_data_dir. Set the parameter transform to the object composed. 
'''
# Create the train dataset and validation dataset

train_dataset = Dataset(transform=composed
                        ,csv_file=train_csv_file
                        ,data_dir=train_data_dir)

validation_dataset = Dataset(transform=composed
                          ,csv_file=validation_csv_file
                          ,data_dir=validation_data_dir)


#Questions-----------------------------------------------------------------------------------
'''
Question 3.1: Preparation
5 points

Step 1: Load the pre-trained model resnet18 Set the parameter pretrained to true.

# Step 1: Load the pre-trained model resnet18

​

# Type your code here

Step 2: The following lines of code will set the attribute requires_grad to False. As a result, the parameters will not be affected by training.

# Step 2: Set the parameter cannot be trained for the pre-trained model

​

# Type your code here

resnet18 is used to classify 1000 different objects; as a result, the last layer has 1000 outputs. The 512 inputs come from the fact that the previously hidden layer has 512 outputs.

Step 3: Replace the output layer model.fc of the neural network with a nn.Linear object, to classify 7 different bills. For the parameters in_features remember the last hidden layer has 512 neurons.

# Step 3: Re-defined the last layer

​

# Type your code here

Print out the model in order to show whether you get the correct answer.
(Your peer reviewer is going to mark based on what you print here.)

# Print the model (PLEASE DO NOT MODIFY THIS BOX)

​

print(model)

Question 3.2: Train the model
5 points
Did you know? IBM Watson Studio lets you build and deploy an AI solution, using the best of open source and IBM software and giving your team a single environment to work in. Learn more here.

Step 1: Create a cross entropy criterion function

# Step 1: Create the loss function

​

# Type your code here

Step 2: Create a training loader and validation loader object, the batch size is 15 and 10 respectively .

# Step 2: Create the data loader

​

# Type your code here

Step 3: Use the following optimizer to minimize the loss

# Step 3: Use the pre-defined optimizer Adam with learning rate 0.003

​

# Type your code here

Step 4: Train the model for 20 epochs, save the loss in a list as will as the accuracy on the validation data for every epoch. The entire process may take 6.5 minutes. Print the validation accuracy for each epoch during the epoch loop. Then, plot the training loss for each epoch and validation error for each epoch.

# Step 4: Train the model

​

N_EPOCHS = 20

loss_list = []

accuracy_list = []

correct = 0

n_test = len(validation_dataset)

​

# Type your code here

Step 5: Plot the training loss for each iteration
(Your peer reviewer is going to mark based on what you plot here.)

# Step 5: Plot the loss for training dataset

​

# Type your code here

Step 6: Plot the validation accuracy for each epoch
(Your peer reviewer is going to mark based on what you plot here.)

# Step 6: Plot the accuracy for valdiation dataset

​

# Type your code here

Question 3.3: Plot 5 Random Images with their predictions
5 points

Create a test dataset using validation data. And, create your own plot_random_image() function to plot 5 random images which index is in the numbers list. Run the function to plot image, print the predicted label and print a string indicate whether it has been correctly classified or mis-classified.
(Your peer reviewer is going to mark based on what you plot here.)

# Plot the images with labels

​

look_up = {0: 'predicted: $5'

           , 1: 'predicted: $10'

           , 2: 'predicted: $20'

           , 3: 'predicted: $50'

           , 4: 'predicted: $100'

           , 5: 'predicted $200'

           , 6: 'predicted $500'}

random.seed(0)

numbers = random.sample(range(70), 5)

​

# Type your code here

Question 3.4: Use the second model Densenet121 to do the prediction
3 points

Repeat the steps in Question 3.1, 3.2 to predict the result using models.densenet121 model. Then, print out the last validation accuracy.

Steps:

    Load the pre-trained model Densenet
    Replace the last classification layer with only 7 classes
    Set the configuration (parameters)
    Train the model
    Print the last validation accuracy

Hint:

    The second last layer for this model has 1024 outputs.
    The last layer for Densenet121 can be accessed by model.classifier
    Use the criterion function nn.CrossEntropyLoss()
    Train Batch Size: 15; Validation Batch Size: 10
    Optimizer: Adam with learning rate 0.003
    10 Epoches. Otherwise, it will take too long.

You are welcome to try any pattern of setting and find out the best result. Please name the model variable as model_des.
(Your peer reviewer is going to mark based on what you print here.)

# Use densenet121 to train the model and print out the last validation accuracy.

​

# Type your code here

Save the trained model

Save the trained model for the following chapters

# Save the model

​

torch.save(model, "resnet18_pytorch.pt")

torch.save(model_des, "densenet121_pytorch.pt")


'''