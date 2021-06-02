from torchvision import datasets,transforms
import torch
import numpy as np
import os

import torch.nn as nn
import torch.nn.functional as F

num_channels = 3  #number of colors on images (B&W=1, RGB=3)
model_save_path = 'models/ptah.pth'
ocr_dir = 'data'

class Net(nn.Module):
    def __init__(self,num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 1000)
        self.fc2 = nn.Linear(1000, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))   
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(net,model_load_path):
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_load_path))
    else:
        net.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
        
    net.eval()

def load_ocr_dataset():
    class_names = np.load('class_names.npy')
    print('Signs in the trained model:',class_names)

    data_transforms = {
            'ocr': transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.Grayscale(num_channels),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),              
        }

    image_datasets = {x: datasets.ImageFolder(os.path.join(ocr_dir, x),
                                        data_transforms[x])
                for x in ['ocr']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                            shuffle=True, num_workers=4)
                for x in ['ocr']}

    print('Sings in the OCR Dataset:',image_datasets['ocr'].classes)
    ocr_classes = image_datasets['ocr'].classes

    return ocr_classes,class_names,dataloaders

def test_ocr_dataset(ocr_dir,model,ocr_classes,class_names,dataloaders):
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['ocr']):
            if torch.cuda.is_available():            
                inputs = inputs.to(model.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            total += labels.size(0)
            l_preds = np.array([class_names[preds[j]] for j in range(len(preds))])
            l_labels = np.array([ocr_classes[labels[j]] for j in range(len(labels))]) 
            correct += (l_preds == l_labels).sum().item()

            print('Ground Truth: ', ' '.join('%5s' % ocr_classes[labels[j]] for j in range(len(labels))))
            print('Predicted   : ', ' '.join('%5s' % class_names[preds[j]] for j in range(len(preds))))        

    print('Accuracy of the network on the test (OCR) images: %d %%' % (
    100 * correct / total)) 

if __name__ == '__main__':    
    ocr_classes,class_names,dataloaders = load_ocr_dataset()

    model = Net(len(class_names))
    load_model(model,model_save_path)

    test_ocr_dataset(ocr_dir,model,ocr_classes,class_names,dataloaders)  