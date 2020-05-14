#Classifying European Money Denominations
#Download the datasets you needed for this lab.-------------------------------------------------
# You can comment the code in this box out if you already have the dataset.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

# Download Training Dataset
!wget --quiet -O /resources/data/training_data_pytorch.tar.gz https://cocl.us/DL0320EN_TRAIN_TAR_PYTORCH
!tar -xzf  /resources/data/training_data_pytorch.tar.gz -C /resources/data --exclude '.*'

# Download Validation Dataset
!wget --quiet -O /resources/data/validation_data_pytorch.tar.gz https://cocl.us/DL0320EN_VALID_TAR_PYTORCH
!tar -xzf  /resources/data/validation_data_pytorch.tar.gz -C /resources/data --exclude '.*'

#The following are the PyTorch modules you are going to need
# PyTorch Modules you need for this lab

from torch.utils.data import Dataset, DataLoader
import pandas
from torchvision import transforms

# Other non-PyTorch Modules

from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image

#Load CSV------------------------------------------------------------------------------------
'''
In this section, you will load the CSV with Pandas. The CSV file contains the name of the image file as well as the class label.

The denomination, file name and the class variable for the training data is stored in the following CSV file.
'''
#Training Data CSV--------------------------------------------------------
# train_csv_file contains the URL that contains the CSV file we needed
train_csv_file = 'https://cocl.us/DL0320EN_TRAIN_CSV'

# Read CSV file from the URL and print out the first five samples
train_data_name = pd.read_csv(train_csv_file)
train_data_name.head()

'''
The first column of the dataframe corresponds to the sample number. The second column is the denomination. 
The third column is the file name, and the final column is the class variable. 
The correspondence between the class variable and each denomination is as follows. 
Five Euros have y equal 0; ten Euros have y equals 1; twenty Euros have y equals 2 and so on.

You can obtain the file name of the first file by using the method  DATAFRAME.iloc[0, 2]. 
The first argument corresponds to the sample number, and the second input corresponds to the column index.
'''
# Get the value on location row 0, column 2 (Notice that index starts at 0.)
print('File name:', train_data_name.iloc[0, 2])

#As the class of the sample is in the fourth row, you can also obtain the class value as follows.
# Get the value on location row 0, column 3 (Notice that index starts at 0.)
print('y:', train_data_name.iloc[0, 3])

#Similarly, You can obtain the file name of the second image file and class number:
# Print out the file name and the class number of the element on row 1 (the second row)
print('File name:', train_data_name.iloc[1, 2])
print('y:', train_data_name.iloc[1, 3])

'''
The number of samples corresponds to the number of rows in a dataframe. You can obtain the number of rows using the following lines of code. 
This will correspond the data attribute len.
'''
# Print out the total number of rows in traing dataset
print('The number of rows: ', train_data_name.shape[0]

#Validation---------------------------------------------------------------------------------------
#We can do the same for the validation data. The data is stored in the following CSV file:
# The url for getting csv file for validation dataset
validation_csv_file='https://cocl.us/DL0320EN_VALID_CSV'

#Try--------------------------------------------------------------
## Load csv file and print the first five rows
validation_data_name = pd.read_csv(validation_csv_file)
validation_data_name.head()

#Load the 11th sample image name and class label
# Print the sample image file name and class number for row 10 (the 11th row)
print("The file name: ", validation_data_name.iloc[10, 2])
print("The class label: ", validation_data_name.iloc[10, 3])

#Load Image-----------------------------------------------------------------------------------------
#The training images are stored in the following directory /resources/data/training_data_pytorch/, you can save it in the variable train_data_dir.
# Save the image folderpath in a variable
train_data_dir = '/resources/data/training_data_pytorch/'

#You can find the file name of a particular image from the Dataframe 
# Print the file name on the second row
train_data_name.iloc[1, 2]

'''
To load the image, you need the directory and the image name. 
You can concatenate the variable train_data_dir with the name of the image stored in a Dateframe. 
Finally, you will store the result in the variable train_image_name
'''
# Combine the directory path with file name
train_image_name = train_data_dir + train_data_name.iloc[1, 2]

#You can then use the function Image.open to store the image to the variable image.
# Plot the second training image

image = Image.open(train_image_name)
plt.imshow(image)
plt.show()

#You can repeat the process for the 20th image.
# Plot the 20th image

train_image_name = train_data_dir + train_data_name.iloc[19, 2]
image = Image.open(train_image_name)
plt.imshow(image)
plt.show()

#Validation Images-------------------------------------------------------------------------------
#The Validation data is stored in the following directory /resources/data/validation_data_pytorch/. You can assign it to the variable validation_data_dir.
# Save the image folderpath in a variable

validation_data_dir='/resources/data/validation_data_pytorch/'

#Create a Dataset Class---------------------------------------------------------------------------
'''
In this section, you will Complete the Dataset object, and the variable names are given . 
This will be a generalization of the above sections. Complete the following Dataset class:
'''
# Create your own dataset object

class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        
        # Image directory
        self.data_dir=data_dir
        
        # The transform is goint to be used on image
        self.transform = transform
        
        # Load the CSV file contians image info
        self.data_name= pd.read_csv(csv_file)
        
        # Number of images in dataset
        self.len=self.data_name.shape[0] 
    
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        
        # Image file path
        img_name=self.data_dir + self.data_name.iloc[idx, 2]
        
        # Open image file
        image = Image.open(img_name)
        
        # The class label for the image
        y = self.data_name.iloc[idx, 3]
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y

'''
Questions

In this section, you will test your dataset object. A dataset object will be created and you will test it by viewing the output.

If your dataset class is correct, the following line of code should work.

If your dataset class is correct, the following line of code should construct a dataset object for the training data and validation data.

# Create the dataset objects

​

train_dataset = Dataset(csv_file=train_csv_file

                        , data_dir='/resources/data/training_data_pytorch/')

validation_data = Dataset(csv_file=validation_csv_file

                          , data_dir='/resources/data/validation_data_pytorch/')

Question 2.1

The following lines of code will print out three sample images and their classes from the training data. Run the code and remember the results; you will be test on the results in the quiz.

# Question 2.1

# Answer the question in the quiz

​

samples = [53, 23, 10]

​

# Type your code here

Question2.2

The following lines of code will print out three sample images and their class from the validation data. Run the code and remember the results; you will be test on the results in the quiz.

# Question 2.2

# Answer the question in the quiz

​

samples =[22, 32, 45]

​

# Type your code here
'''
#Test Transform---------------------------------------------------------------------
'''
Use the constructor compose to perform the following sequence of transformations in the order they are given. 
Then test your dataset class to see if the transform is implemented correctly.
'''
# Create the transform compose

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])

# Create a test_normalization dataset using composed as transform
test_normalization = Dataset(csv_file=train_csv_file
                        , data_dir='/resources/data/training_data_pytorch/'
                        , transform = composed)

# Print mean and std

print("Mean: ", test_normalization[0][0].mean(dim = 1).mean(dim = 1))
print("Std:", test_normalization[0][0].std(dim = 1).std(dim = 1))