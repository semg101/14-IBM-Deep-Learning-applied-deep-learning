#Classifying European Money Denominations: Comparing Two Models
#Preparation-------------------------------------------------------------------
# You can comment out this box when you already have the dataset
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

# Download test dataset
!wget --quiet -O /resources/data/test_data_pytorch.tar.gz https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0320EN/Datasets/PyTorch/test_data_pytorch.tar.gz
!tar -xzf /resources/data/test_data_pytorch.tar.gz -C /resources/data --exclude '.*'

# Import PyTorch Modules that will be used in the lab
import torch 
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas
from torchvision import transforms
import torch.nn as nn
torch.manual_seed(0)

# Import Non-PyTorch Modules that will be used in the lab
import time
from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

#Create Dataset Class and Object-----------------------------------------------------------
# Url that contains CSV files and image dataset folder
test_csv_file = 'https://cocl.us/DL0320EN_TEST_CSV'
test_data_dir = '/resources/data/test_data_pytorch/'

# Create Dataset Class

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

#try----------------------------------------------------------
# Construct the composed object for transforming the image 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
trans_step = [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)]

composed = transforms.Compose(trans_step)

'''
Create a test dataset object using the CSV file stored in the variables the test_csv_file. 
The directories are stored in the variable test_data_dir. Set the parameter transform to the object composed. 
'''
# Create a test_dataset
test_dataset = Dataset(transform=composed
                       , csv_file=test_csv_file
                       , data_dir=test_data_dir)

#Load Pre-trained Model---------------------------------------------------------
#Load the ResNet18 and Densenet121 model you created from the last section 
# Load pre-trained model
model = torch.load("resnet18_pytorch.pt")
model_des = torch.load("densenet121_pytorch.pt")

# Print model structure
print("ResNet18:\n", model)
print("Densenet121:\n", model_des)

# Set Data Loader object
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10)

#Analyze Models----------------------------------------------------------------------------
#try----------------------------------------
#Find the error on the test data using the ResNet18 model.
# Predict the data using ResNet18 model and print out accuracy
correct = 0
accuracy = 0
N = len(test_dataset)
for x_test, y_test in test_loader:
    model.eval()
    z = model(x_test)
    _, yhat = torch.max(z.data, 1)
    correct += (yhat == y_test).sum().item()
accuracy = correct / N
print("Accuracy using ResNet18: ", accuracy) 

#try--------------------------------------------------
#Find the error on the test data using the Densenet121 model
# Predict the data using densene model and print out accuracy
correct = 0
accuracy = 0
N = len(test_dataset)
for x_test, y_test in test_loader:
    model_des.eval()
    z = model_des(x_test)
    _, yhat = torch.max(z.data, 1)
    correct += (yhat == y_test).sum().item()
accuracy = correct / N
print("Accuracy using Densenet121: ", accuracy) 

