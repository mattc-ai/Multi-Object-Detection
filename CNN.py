import os
import shutil
from skimage import io, transform
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Crop2WeedDataset(Dataset):

    def __init__(self, csv_dir, img_dir, transform=None, test=True):
        self.csv_dir = csv_dir
        self.img_dir = img_dir
        self.transform = transform
        name_file = open("test_split.txt", 'r') if test else open("train_split.txt", 'r')
        self.names = name_file.readlines()
        self.img_width = 1920
        self.img_height = 1088
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        name = self.names[idx].rstrip('\n')

        img_name = os.path.join(self.img_dir, name + ".jpg")
        image = io.imread(img_name)
        data_file = open(os.path.join(self.csv_dir, name + ".csv"), 'r')
        data = data_file.read().rstrip('\n').split(',')
        box = [int(data[0])/self.img_width, int(data[1])/self.img_height, int(data[2])/self.img_width, int(data[3])/self.img_height]
        box = np.array(box, dtype=np.float32)
        cl = data[4]
        sample = {'image': image, 'box': box, 'class': cl}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

#Images resolution 1920x1088x3
class MOD(nn.Module):
    def __init__(self):
        super(MOD,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3) 
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        #self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        #self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.flatten = nn.Flatten()  #make multi-dimensional input one-dimensional

        #Bounding box layers
        bbox_input_size = 39*70*64 #height, width, channel
        self.bbox1 = nn.Linear(in_features=bbox_input_size, out_features=128)
        self.bbox2 = nn.Linear(in_features=128, out_features=64)
        self.bbox3 = nn.Linear(in_features=64, out_features=32)
        self.bbox4 = nn.Linear(in_features=32, out_features=4) 
        self.sigmoid = nn.Sigmoid()
        
        #Classification layers
        #cl_input_size = 136*240*20 #height, width, channel
        #self.cl1 = nn.Linear(in_features=cl_input_size, out_features=120)
        #self.cl3 = nn.Linear(in_features=120, out_features=1) #binary 0 or 1


    def forward(self,x):
        #[10, 3, 1088, 1920]
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        #[10, 16, 362, 639]
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        #[10, 32, 120, 212]
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        #[10, 64, 39, 70]
        x = self.flatten(x)
        #[10, 174720]

        #Bounding Box
        x_bbox = self.bbox1(x)
        x_bbox = self.bbox2(x_bbox)
        x_bbox = self.bbox3(x_bbox)
        x_bbox = self.sigmoid(x_bbox)
        x_bbox = self.bbox4(x_bbox)
       
        #Classification
        #x_class = self.cl1(x)
        #x_class = F.leaky_relu(x_class)
        #x_class = self.cl3(x_class)

        return x_bbox
classes = ["ave", "vwg"]

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda') # or cuda for windows



#organize_dataset("cropandweed-dataset/data/images", "subdivided_images/")
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
])
image_data = Crop2WeedDataset("cropandweed-dataset/data/bboxes/CropOrWeed2/", "cropandweed-dataset/data/images/", test=False, transform=transform)
#image_data = torchvision.datasets.ImageFolder("subdivided_images/", transform=transform)
trainloader = torch.utils.data.DataLoader(image_data, batch_size=10, shuffle=True)

model = MOD().to(device)

def train():
    criterion = nn.MSELoss() # Defining the criterion
    optimizer = optim.Adam(model.parameters(), lr=0.0001)# Defining the optimizer

    for epoch in range(1):  # Looping over the dataset three times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs = data["image"] # The input data is a list [inputs, labels]
            labels = data["box"]
            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # Setting the parameter gradients to zero
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels)# Applying the criterion
            loss.backward() # Backward pass
            optimizer.step()  # Optimization step

            running_loss += loss.item() # Updating the running loss

            print('[epoch: %d, mini-batch: %d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

        torch.save(model.state_dict(), "state/model.pth")

    print('Finished Training')

def test():
    model.load_state_dict(torch.load("state/model.pth"))
    dataiter = iter(trainloader) # Iterator over the testing set
    data = next(dataiter)
    inputs = data["image"] # The input data is a list [inputs, labels]
    labels = data["box"]

    model.cpu()

    # Printing the ground truth images and labels
    print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(len(labels))))

    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    # Printing the labels the network assigned to the input images
    print('\nPredicted:   ', ' '.join('%5s' % outputs[j] for j in range(len(labels))))


train()
test()