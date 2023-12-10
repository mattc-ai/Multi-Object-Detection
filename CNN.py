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
import pandas as pd

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
        cl = np.array(int(data[4]), dtype=np.float32).reshape(1)
        sample = {'image': image, 'box': box, 'class': cl}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

#Images resolution 1920x1088x3
class MOD(nn.Module):
    def __init__(self):
        super(MOD,self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        # Bounding box prediction head
        self.bbox_head = nn.Sequential(
            nn.Linear(32 * 10 * 18, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU()
        )

        self.classification_head = nn.Sequential(
            nn.Linear(32 * 10 * 18, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )


    def forward(self,x):
        x = self.backbone(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        bbox_predictions = self.bbox_head(x)
        classification_output = self.classification_head(x)
        return bbox_predictions, classification_output

classes = ["ave", "vwg"]

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda') # or cuda for windows



#organize_dataset("cropandweed-dataset/data/images", "subdivided_images/")
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
])
image_data = Crop2WeedDataset("cropandweed-dataset/data/bboxes/CropOrWeed2/", "cropandweed-dataset/data/images/", test=False, transform=transform)
#image_data = torchvision.datasets.ImageFolder("subdivided_images/", transform=transform)
trainloader = torch.utils.data.DataLoader(image_data, batch_size=32, shuffle=True)

model = MOD().to(device)
#model.load_state_dict(torch.load("state/model.pth"))

def train():
    num_epoch = 10
    bbox_criterion = nn.MSELoss(reduction='sum')
    class_criterion = nn.BCELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.001)# Defining the optimizer
    scheduler_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1, total_iters=num_epoch)
    mean_loss = []

    for epoch in range(num_epoch):  # Looping over the dataset three times
        loss_df = pd.DataFrame(index=list(range(len(trainloader))))

        offset = (epoch * len(trainloader))
        for i, data in enumerate(trainloader, 0):

            inputs = data["image"]
            bbox_label = data["box"]
            class_label = data["class"]
            inputs, bbox_label, class_label = inputs.to(device), bbox_label.to(device), class_label.to(device)

            optimizer.zero_grad()
            
            bbox_out, class_out = model(inputs) # Forward pass
            print(class_out)

            bbox_loss = bbox_criterion(bbox_out, bbox_label)
            class_loss = class_criterion(class_out, class_label)
            total_loss = bbox_loss  + class_loss 

            loss_df.at[offset+i, "loss"] = total_loss.item()# if loss.item() < 10 else 1
            total_loss.backward() # Backward pass
            optimizer.step()  # Optimization step

            print('[epoch: %d, mini-batch: %d] loss_bbox: %.3f, loss_class: %.3f' %
                    (epoch + 1, i + 1, bbox_loss.item(), class_loss.item()))
        
        if epoch != 0:
            mean_loss.append(loss_df.mean()) 
            xs = pd.DataFrame(mean_loss).plot()
            fig = xs.get_figure()
            fig.savefig('figure_' + str(epoch) + '.png')
            print(mean_loss)
        
        print("Learing rate:" + str(optimizer.param_groups[0]["lr"]))
        scheduler_lr.step()
        #test()

        torch.save(model.state_dict(), "state/model.pth")

    print('Finished Training')

def test():
    model.load_state_dict(torch.load("state/model.pth"))
    dataiter = iter(trainloader) # Iterator over the testing set
    data = next(dataiter)
    inputs = data["image"]
    bbox_label = data["box"]
    class_label = data["class"]

    model.cpu()

    bbox_out, class_out = model(inputs)

    print(bbox_out)

    for x_bbox, x_class, y_bbox, y_class in zip(bbox_label, class_label, bbox_out, class_out):
        print(x_bbox, x_class, y_bbox, y_class)



train()
test()