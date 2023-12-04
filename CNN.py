import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#Images resolution 1920x1088x3
class MOD(nn.Module):
    def __init__(self):
        super(MOD,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()  #make multi-dimensional input one-dimensional

        #Bounding box layers
        bbox_input_size = 136*240*512 #height, width, channel
        self.bbox1 = nn.Linear(in_features=bbox_input_size, out_features=256)
        self.bbox2 = nn.Linear(in_features=256, out_features=4) #bbox params x, y, w, h
        
        #Classification layers
        cl_input_size = 136*240*512 #height, width, channel
        self.cl1 = nn.Linear(in_features=cl_input_size, out_features=4096)
        self.cl2 = nn.Linear(in_features=4096, out_features=4096)
        self.cl3 = nn.Linear(in_features=4096, out_features=1) #binary 0 or 1

    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.pool2
        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = self.conv5(x)
        x = F.leaky_relu(x)
        x = self.pool3(x)
        x = self.flatten(x)

        #Bounding Box
        x_bbox = self.bbox1(x)
        x_bbox = F.leaky_relu(x_bbox)
        x_bbox = self.bbox2(x_bbox)
       
        #Classification
        x_class = self.cl1(x)
        x_class = F.leaky_relu(x_class)
        x_class = self.cl2(x_class)
        x_class = F.leaky_relu(x_class)
        x_class = self.cl3(x_class)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('mps') # or cuda for windows