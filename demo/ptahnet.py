import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler


import torchvision
from torchvision import datasets, models, transforms

import torch

import numpy as np
import os
import time
import copy

# import PIL
from PIL import Image

class Net(nn.Module):
    def __init__(self,num_channels,data_dir,mode="training"):
        super(Net, self).__init__()

        self.num_channels = num_channels
        self.data_dir = data_dir
        self.mode = mode
        self.num_letters = self._loaddata()

        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 1000)
        self.fc2 = nn.Linear(1000, 128)
        self.fc3 = nn.Linear(128, self.num_letters)

        self._prepare_device()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))   
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _prepare_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ",device)
        self.device = device

        if torch.cuda.is_available():
            self.cuda()

    def save_model(self,model_save_path):
        torch.save(self.state_dict(), model_save_path)
        self.model_save_path = model_save_path

    def load_model(self,model_load_path):
        self.load_state_dict(torch.load(model_load_path))
        self.eval()

    def train_model(self,num_epochs,learning_rate):
        best_acc = 0.0
        since = time.time()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            print('-' * 40)
            print('Epoch {}/{} : LR:{}'.format(epoch, num_epochs - 1,scheduler.get_last_lr()))
            print('-' * 40)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.train()  # Set model to training mode
                else:
                    self.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.6f} Acc: {:.6f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.state_dict())

        #         print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('-----------------------------------------------------------')
        print("Epochs:",num_epochs)
        print("Learning Rate:",learning_rate)        

    def _loaddata(self):


        data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize([32, 32]),        
                    transforms.Grayscale(self.num_channels),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize([32, 32]),
                    transforms.Grayscale(self.num_channels),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': transforms.Compose([
                    transforms.Resize([32, 32]),
                    transforms.Grayscale(self.num_channels),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),  
                'full_test': transforms.Compose([
                    transforms.Resize([32, 32]),
                    transforms.Grayscale(self.num_channels),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'ocr': transforms.Compose([
                    transforms.Resize([32, 32]),
                    transforms.Grayscale(self.num_channels),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),              
            }

        self.data_transforms = data_transforms

        if self.mode == "inference":
            print('Inference: loading classes from file')
            class_names = np.load('class_names.npy')
            # class_names = np.fromfile("class_names")
            # print (class_names)
            self.class_names = class_names
            return len(class_names)

        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                data_transforms[x])
                        for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                    shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        class_names = image_datasets['train'].classes

        print('Classes loaded:',class_names)

        num_letters = len(class_names)

        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.class_names = class_names

        # Save class names for inference (so we do not need to load dataset)
        np.save("class_names", np.array(class_names))

        return len(class_names)


    def validate_dataset(self,loader):
        self.eval()
        
        correct = 0
        total = 0

        wrong_results = np.zeros(self.num_letters)
        true_samples= np.zeros(self.num_letters)
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self(inputs)
                _, preds = torch.max(outputs,1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                for j in range(0,len(preds.cpu().numpy())):
                    pred = preds.cpu().numpy()[j]
                    truth = labels.cpu().numpy()[j]
                    if pred != truth:
                        wrong_results[truth] = wrong_results[truth] + 1
                    true_samples[truth] = true_samples[truth] + 1
            print('Accuracy per label:',(true_samples-wrong_results)/true_samples)

            return correct,total


