# import ptahnet as pn

import os
import time
import copy
import torch
from torchvision import datasets,transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import ptah_net

#number of colors on images (B&W=1, RGB=3)
num_channels = 3
num_epochs = 50
learning_rate = 0.0009978380594782084
model_save_path = 'models/ptah.pth'
data_dir = "../data"

def load_data():
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([32, 32]),        
                transforms.Grayscale(num_channels),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.Grayscale(num_channels),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),             
        }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    print('Classes loaded:',class_names)

    # Save class names for inference (so we do not need to load dataset)
    np.save("class_names", np.array(class_names))

    return class_names,dataloaders,data_transforms,dataset_sizes

def train_model(net,dataloaders,dataset_sizes):
        best_acc = 0.0
        since = time.time()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Training on: ',device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            print('-' * 40)
            print('Epoch {}/{} : LR:{}'.format(epoch, num_epochs - 1,scheduler.get_last_lr()))
            print('-' * 40)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    net.train()  # Set model to training mode
                else:
                    net.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = net(inputs)
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

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.6f} Acc: {:.6f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(net.state_dict())

        #         print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('----------------------------------------')
        print("Epochs:",num_epochs)
        print("Learning Rate:",learning_rate) 

def validate_dataset(net,loader,num_letters):
    net.eval()
    
    correct = 0
    total = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wrong_results = np.zeros(num_letters)
    true_samples= np.zeros(num_letters)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
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

        print('Accuracy of the network on the Validation Dataset: %d %%' % (
                    100 * correct / total))


if __name__ == '__main__':    
    class_names,dataloaders,data_transforms,dataset_sizes = load_data()

    net = ptah_net.PtahNet(len(class_names))

    if torch.cuda.is_available():
        net.cuda()

    train_model(net,dataloaders,dataset_sizes)

    torch.save(net.state_dict(), model_save_path)

    validate_dataset(net,dataloaders['val'],len(class_names))
