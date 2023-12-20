import os
import shutil
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms
import pandas as pd
from random import randint

MAX_DETECTION = 206
IMG_WIDTH = 1920
IMG_HEIGHT = 1088
BATCH_SIZE = 1

class Crop2WeedDataset(Dataset):

    def __init__(self, csv_dir, img_dir, transform=None, test=True):
        self.csv_dir = csv_dir
        self.img_dir = img_dir
        self.transform = transform
        name_file = open("test_split.txt", 'r') if test else open("train_split.txt", 'r')
        self.names = name_file.readlines()
        self.img_width = IMG_WIDTH
        self.img_height = IMG_HEIGHT
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #get the name of the file
        name = self.names[idx].rstrip('\n')
        #image
        img_name = os.path.join(self.img_dir, name + ".jpg")
        image = Image.open(img_name)
        #label
        data_file = open(os.path.join(self.csv_dir, name + ".csv"), 'r')
        data = data_file.read().replace('\n', ',').split(',')
        #label normalization
        box_list = [0] * (MAX_DETECTION * 4)
        class_list = [0] * MAX_DETECTION
        box_i = 0
        class_i = 0
        count = 0
        while count < len(data)-1:
            # label in range [0, 1]
            box_list[box_i] = int(data[count])/self.img_width
            box_list[box_i+1] = int(data[count+1])/self.img_height
            box_list[box_i+2] = int(data[count+2])/self.img_width
            box_list[box_i+3] = int(data[count+3])/self.img_height
            class_list[class_i] = int(data[count+4])
            class_i += 1
            box_i += 4
            count += 7
        
        sample = {}

        if self.transform:
            sample["image"] = self.transform(image)
        
        sample["bboxes"] = torch.tensor(box_list)
        sample["class"] = torch.tensor(class_list, dtype=torch.float)

        return sample

class MOD(nn.Module):
    def __init__(self, max_output):
        super(MOD,self).__init__()

        #cnn
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        #Bounding box prediction
        self.bbox_head = nn.Sequential(
            nn.Linear(32 * 10 * 18, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, max_output*4)
        )

        #Classification prediction
        self.classification_head = nn.Sequential(
            nn.Linear(32 * 10 * 18, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, max_output),
            nn.Sigmoid()
        )


    def forward(self,x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        bbox_predictions = self.bbox_head(x)
        classification_output = self.classification_head(x)
        return bbox_predictions, classification_output

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale()
])

train_data = Crop2WeedDataset("cropandweed-dataset/data/bboxes/CropOrWeed2/", "cropandweed-dataset/data/images/", test=False, transform=transform)
test_data = Crop2WeedDataset("cropandweed-dataset/data/bboxes/CropOrWeed2/", "cropandweed-dataset/data/images/", test=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

model = MOD(MAX_DETECTION).to(device)
model.load_state_dict(torch.load("state/model_11.pth"))

def train():
    #defining epochs, loss functions and optimizer 
    num_epoch = 12
    bbox_criterion = nn.MSELoss(reduction='sum')
    class_criterion = nn.BCELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=num_epoch)

    mean_train = []
    mean_test = pd.DataFrame()

    for epoch in range(num_epoch):
        loss_df = pd.DataFrame()

        for i, data in enumerate(trainloader):

            images = data["image"].to(device)
            bbox_label = data["bboxes"].to(device)
            class_label = data["class"].to(device)

            optimizer.zero_grad()
            
            bbox_out, class_out = model(images)
            
            bbox_loss = bbox_criterion(bbox_out, bbox_label)
            class_loss = class_criterion(class_out, class_label)
            total_loss = bbox_loss + class_loss 

            loss_df.at[i, "loss"] = total_loss.item()
            total_loss.backward()
            optimizer.step()

            print('[epoch: %d, mini-batch: %d] %.3f + %.3f = %.3f' %
                    (epoch + 1, i + 1, bbox_loss.item(), class_loss.item(), total_loss.item()))
        
        #not considering the first epoch because it's usually out of scale
        if epoch != 0:
            #calculating and printing the mean of the current epoch loss function
            mean_train.append(loss_df.mean()) 
            xs = pd.DataFrame(mean_train).plot()
            fig = xs.get_figure()
            fig.savefig('train_loss.png')
            print(mean_train)
            #calculating and printing the mean of test set loss function
            mean_test.at[epoch, "loss"] = test().item()
            xs = mean_test.plot()
            fig = xs.get_figure()
            fig.savefig('test_loss.png')


        torch.save(model.state_dict(), "state/model_" + str(epoch) + ".pth")

        print("Learing rate:" + str(optimizer.param_groups[0]["lr"]))
        scheduler_lr.step()


    print('Finished Training')

def test(display=False):

    bbox_criterion = nn.MSELoss(reduction='sum')
    class_criterion = nn.BCELoss(reduction='sum')

    dataiter = iter(testloader)
    data = next(dataiter)
    
    images = data["image"]
    bbox_label = data["bboxes"]
    class_label = data["class"]
    
    model.cpu()

    bbox_out, class_out = model(images)

    #To print model results
    #for x_bbox, x_class, y_bbox, y_class in zip(bbox_label, class_label, bbox_out, class_out):
    #    print(x_bbox, x_class, y_bbox, y_class)

    #To print images with box
    if display:
        #going from range [0, 1] to [0, img_dim]
        norm_bbox_label = []
        for batch in bbox_label.reshape(BATCH_SIZE, MAX_DETECTION, 4):
            for box in batch:
                norm_bbox_label.append([box[0]*IMG_WIDTH, box[1]*IMG_HEIGHT, box[2]*IMG_WIDTH, box[3]*IMG_HEIGHT])
        
        norm_bbox_out = []
        for i, batch in enumerate(bbox_out.reshape(BATCH_SIZE, MAX_DETECTION, 4)):
            norm_bbox_out.append([])
            for box in batch:
                boxes = torchvision.ops.box_convert(
                    torch.tensor([abs(box[0]*IMG_WIDTH), abs(box[1]*IMG_HEIGHT), abs(box[2]*IMG_WIDTH), abs(box[3]*IMG_HEIGHT)]), "xyxy", "xyxy")
                norm_bbox_out[i].append([
                    boxes[0] if boxes[2] < boxes[0] else boxes[2],
                    boxes[1] if boxes[1] < boxes[3] else boxes[3],
                    boxes[2] if boxes[2] > boxes[0] else boxes[0],
                    boxes[3] if boxes[2] > boxes[0] else boxes[1]
                ])
            
        norm_bbox_label = torch.tensor(norm_bbox_label).reshape(BATCH_SIZE, MAX_DETECTION, 4)
        norm_bbox_out = torch.tensor(norm_bbox_out).reshape(BATCH_SIZE, MAX_DETECTION, 4)

        for i, (image, label, predict) in enumerate(zip(images, norm_bbox_label, norm_bbox_out)):
            Image.fromarray(
                np.array(
                    draw_bounding_boxes(
                        torch.tensor(np.array(image*255, dtype=np.uint8)), 
                        label, 
                        width=4,
                        colors="white"
                        )
                    )
                [0]
            ).save("label" + str(i) +".png")
            Image.fromarray(
                np.array(
                    draw_bounding_boxes(
                        torch.tensor(np.array(image*255, dtype=np.uint8)), 
                        predict, 
                        width=4,
                        colors="white"
                        )
                    )
                [0]
            ).save("output" + str(i) +".png")
    
    bbox_loss = bbox_criterion(bbox_out.reshape(-1), bbox_label.reshape(-1))
    class_loss = class_criterion(class_out, class_label)
    model.to(device)
    
    return bbox_loss + class_loss

train()