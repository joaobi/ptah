import ptahnet as pn

# import torchvision
from torchvision import datasets
import torch
import numpy as np
import os


#number of colors on images (B&W=1, RGB=3)
num_channels = 3
num_epochs = 1
learning_rate = 0.0009978380594782084
model_save_path = 'models/ptah.pth'

def load_model(net,model_load_path):
    net.load_state_dict(torch.load(model_load_path))
    net.eval()

def test_ocr_dataset(ocr_dir):
    correct = 0
    total = 0
    class_names = np.load('class_names.npy')

    print (class_names)

    image_datasets = {x: datasets.ImageFolder(os.path.join(ocr_dir, x),
                                        net.data_transforms[x])
                for x in ['ocr']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                            shuffle=True, num_workers=4)
                for x in ['ocr']}

    print('Classes in OCR Dataset:',image_datasets['ocr'].classes)
    ocr_classes = image_datasets['ocr'].classes

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['ocr']):
            inputs = inputs.to(net.device)
            # labels = labels.to(self.device)
            outputs = net(inputs)
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

    net = pn.Net(num_channels,"../data","inference")

    load_model(net,model_save_path)

    net.test_ocr_dataset('data')  