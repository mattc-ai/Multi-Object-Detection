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
BATCH_SIZE = 32

class Crop2WeedDataset(Dataset):

    def __init__(self, csv_dir, img_dir, transform=None, test=True, rand_flip=False):
        self.csv_dir = csv_dir
        self.img_dir = img_dir
        self.transform = transform
        name_file = open("test_split.txt", 'r') if test else open("train_split.txt", 'r')
        self.names = name_file.readlines()
        self.img_width = IMG_WIDTH
        self.img_height = IMG_HEIGHT
        self.rand_flip = rand_flip
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        name = self.names[idx].rstrip('\n')

        img_name = os.path.join(self.img_dir, name + ".jpg")
        image = Image.open(img_name)
        data_file = open(os.path.join(self.csv_dir, name + ".csv"), 'r')
        data = data_file.read().replace('\n', ',').split(',')

        box_list = [0] * (MAX_DETECTION * 4)
        class_list = [0] * MAX_DETECTION
        box_i = 0
        class_i = 0
        count = 0
        while count < len(data)-1:
            box_list[box_i] = int(data[count])/self.img_width
            box_list[box_i+1] = int(data[count+1])/self.img_height
            box_list[box_i+2] = int(data[count+2])/self.img_width
            box_list[box_i+3] = int(data[count+3])/self.img_height
            class_list[class_i] = int(data[count+4])
            class_i += 1
            box_i += 4
            count += 7
        
        sample = {}

        rand = randint(0,1) if self.rand_flip else 0

        if self.transform and rand == 1:
            sample["image"] = torch.flip(self.transform(image), [0, 1])
            swap = box[1]
            box[1] = 1 - box[3]
            box[3] = 1 - swap
            sample["box"] = torch.tensor(box)
        elif self.transform:
            sample["image"] = self.transform(image)
            sample["bboxes"] = torch.tensor(box_list).reshape(MAX_DETECTION, 4)
        
        sample["class"] = np.array(class_list, dtype=np.float32)

        #norm_bbox = []
        #for i, box in enumerate(sample["bboxes"]):
        #    if i % 2 == 0:
        #        norm_bbox.append(box * self.img_width)
        #    else:
        #        norm_bbox.append(box * self.img_height)
        #box = torch.tensor(norm_bbox)
        #Image.fromarray(np.array(draw_bounding_boxes(torch.tensor(np.array(sample["image"]*255, dtype=np.uint8)), box.reshape(MAX_DETECTION, 4)))[0]).save("test" + str(idx) +".png")

        return sample

#Images resolution 1920x1088x3
class MOD(nn.Module):
    def __init__(self, max_output):
        super(MOD,self).__init__()
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

        # Bounding box prediction head
        self.bbox_head = nn.Sequential(
            nn.Linear(32 * 10 * 18, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, max_output*4)
        )

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
    transforms.Grayscale()
])
train_data = Crop2WeedDataset("cropandweed-dataset/data/bboxes/CropOrWeed2/", "cropandweed-dataset/data/images/", test=False, transform=transform, rand_flip=False)
test_data = Crop2WeedDataset("cropandweed-dataset/data/bboxes/CropOrWeed2/", "cropandweed-dataset/data/images/", test=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

model = MOD(MAX_DETECTION).to(device)
model.load_state_dict(torch.load("state/model_start.pth"))

def train():
    num_epoch = 12
    bbox_criterion = nn.MSELoss(reduction='sum')
    class_criterion = nn.BCELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.00035)# Defining the optimizer
    scheduler_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=num_epoch)
    mean_loss = []
    mean_test = pd.DataFrame(index=list(range(num_epoch)))

    for epoch in range(num_epoch):  # Looping over the dataset three times
        loss_df = pd.DataFrame(index=list(range(len(trainloader))))

        offset = (epoch * len(trainloader))
        for i, data in enumerate(trainloader, 0):

            inputs = data["image"]
            bbox_label = data["bboxes"]
            class_label = data["class"]
            inputs, bbox_label, class_label = inputs.to(device), bbox_label.to(device), class_label.to(device)

            optimizer.zero_grad()
            
            bbox_out, class_out = model(inputs) # Forward pass
            

            bbox_loss = bbox_criterion(bbox_out.reshape(BATCH_SIZE if i != 192 else 20, MAX_DETECTION, 4), bbox_label)
            class_loss = class_criterion(class_out, class_label)
            total_loss = bbox_loss  + class_loss 

            loss_df.at[offset+i, "loss"] = total_loss.item()# if loss.item() < 10 else 1
            total_loss.backward() # Backward pass
            optimizer.step()  # Optimization step

            print('[epoch: %d, mini-batch: %d] %.3f + %.3f = %.3f' %
                    (epoch + 1, i + 1, bbox_loss.item(), class_loss.item(), total_loss.item()))
        
        if epoch != 0:
            mean_loss.append(loss_df.mean()) 
            xs = pd.DataFrame(mean_loss).plot()
            fig = xs.get_figure()
            fig.savefig('train_loss.png')
            print(mean_loss)
            mean_test.at[epoch, "loss"] = test().item()
            xs = mean_test.plot()
            fig = xs.get_figure()
            fig.savefig('test_loss.png')


        torch.save(model.state_dict(), "state/model_" + str(epoch) + ".pth")

        print("Learing rate:" + str(optimizer.param_groups[0]["lr"]))
        scheduler_lr.step()


    print('Finished Training')

def test():
    #model.load_state_dict(torch.load("state/model_11.pth"))
    bbox_criterion = nn.MSELoss(reduction='sum')
    class_criterion = nn.BCELoss(reduction='sum')
    dataiter = iter(testloader)
    data = next(dataiter)
    inputs = data["image"]
    bbox_label = data["bboxes"]
    class_label = data["class"]
    #inputs, bbox_label, class_label = inputs.to(device), bbox_label.to(device), class_label.to(device)
    model.cpu()

    bbox_out, class_out = model(inputs)

    #for x_bbox, x_class, y_bbox, y_class in zip(bbox_label, class_label, bbox_out, class_out):
    #    print(x_bbox, x_class, y_bbox, y_class)

    #To print images with box
    '''
    norm_bbox_label = []
    for batch in bbox_label:
        for box in batch:
            norm_bbox_label.append([box[0]*IMG_WIDTH, box[1]*IMG_HEIGHT, box[2]*IMG_WIDTH, box[3]*IMG_HEIGHT])
    
    norm_bbox_out = []
    for i, batch in enumerate(bbox_out.reshape(BATCH_SIZE, MAX_DETECTION*4)):
        norm_bbox_out.append([])
        for box in batch.reshape(MAX_DETECTION, 4):
            boxes = torchvision.ops.box_convert(torch.tensor([box[0]*IMG_WIDTH if box[0] > 0 else 0, box[1]*IMG_HEIGHT if box[1] > 0 else 0, box[2]*IMG_WIDTH if box[2] > 0 else 0, box[3]*IMG_HEIGHT if box[3] > 0 else 0]), "xyxy", "xyxy")
            norm_bbox_out[i].append([
                boxes[0] if boxes[2] < boxes[0] else boxes[2],
                boxes[1] if boxes[1] < boxes[3] else boxes[3],
                boxes[2] if boxes[2] > boxes[0] else boxes[0],
                boxes[3] if boxes[2] > boxes[0] else boxes[1]
                ])
        

    box_label = torch.tensor(norm_bbox_label).reshape(BATCH_SIZE, MAX_DETECTION*4)

    for i, image in enumerate(inputs):
        Image.fromarray(
            np.array(
                draw_bounding_boxes(
                    torch.tensor(np.array(image*255, dtype=np.uint8)), 
                    box_label[i].reshape(MAX_DETECTION,4), 
                    width=4,
                    colors="orange"
                    )
                )
            [0]
        ).save("label" + str(i) +".png")
        Image.fromarray(
            np.array(
                draw_bounding_boxes(
                    torch.tensor(np.array(image*255, dtype=np.uint8)), 
                    torch.tensor(norm_bbox_out[i]).reshape(MAX_DETECTION,4), 
                    width=4,colors="orange"
                    )
                )
            [0]
        ).save("output" + str(i) +".png")
    '''
    bbox_loss = bbox_criterion(bbox_out.reshape(BATCH_SIZE, MAX_DETECTION, 4), bbox_label)
    class_loss = class_criterion(class_out, class_label)
    model.to(device)
    
    return bbox_loss + class_loss

train()