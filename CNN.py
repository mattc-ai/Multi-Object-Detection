import os
import shutil
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def organize_dataset(input_path, output_path):

    if os.path.exists(output_path):
        return
    
    os.makedirs(output_path)


    for filename in os.listdir(input_path):
        # Extract the first three letters of the filename
        class_name = filename[:3]

        # Create a subfolder for the class in the output path
        class_path = os.path.join(output_path, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)

        # Move the file to the corresponding class subfolder
        src_filepath = os.path.join(input_path, filename)
        dst_filepath = os.path.join(class_path, filename)
        shutil.move(src_filepath, dst_filepath)

#Images resolution 1920x1088x3
class MOD(nn.Module):
    def __init__(self):
        super(MOD,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        #self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.flatten = nn.Flatten()  #make multi-dimensional input one-dimensional

        #Bounding box layers
        bbox_input_size = 136*240*20 #height, width, channel
        self.bbox1 = nn.Linear(in_features=bbox_input_size, out_features=256)
        self.bbox2 = nn.Linear(in_features=256, out_features=4) #bbox params x, y, w, h
        
        #Classification layers
        cl_input_size = 136*240*20 #height, width, channel
        self.cl1 = nn.Linear(in_features=cl_input_size, out_features=120)
        self.cl3 = nn.Linear(in_features=120, out_features=1) #binary 0 or 1

    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.flatten(x)

        #Bounding Box
        x_bbox = self.bbox1(x)
        x_bbox = F.leaky_relu(x_bbox)
        x_bbox = self.bbox2(x_bbox)
       
        #Classification
        x_class = self.cl1(x)
        x_class = F.leaky_relu(x_class)
        x_class = self.cl3(x_class)

        return [x_bbox, x_class]

classes = ["ave", "vwg"]

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda') # or cuda for windows

organize_dataset("cropandweed-dataset/data/images", "subdivided_images/")
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])
image_data = torchvision.datasets.ImageFolder("subdivided_images/", transform=transform)
trainloader = torch.utils.data.DataLoader(image_data, batch_size=16, shuffle=True)

model = MOD().to(device)

def train():

    criterion = nn.CrossEntropyLoss() # Defining the criterion
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)# Defining the optimizer

    for epoch in range(1):  # Looping over the dataset three times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data # The input data is a list [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # Setting the parameter gradients to zero
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs[0], labels)# Applying the criterion
            loss.backward() # Backward pass
            optimizer.step()  # Optimization step

            running_loss += loss.item() # Updating the running loss

            print('[epoch: %d, mini-batch: %d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), "state/model.txt")

def test():
    model.load_state_dict(torch.load("state/model.txt"))
    dataiter = iter(trainloader) # Iterator over the testing set
    images, labels = next(dataiter)
    model.cpu()

    # Printing the ground truth images and labels
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

    outputs = model(images)
    _, predicted = torch.max(outputs[0], 1)

    # Printing the labels the network assigned to the input images
    print('Predicted:   ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(len(labels))))

test()