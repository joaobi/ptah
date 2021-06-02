import ptahnet as pn

#number of colors on images (B&W=1, RGB=3)
num_channels = 3
num_epochs = 50
learning_rate = 0.0009978380594782084
model_save_path = 'models/ptah.pth'





if __name__ == '__main__':    

    net = pn.Net(num_channels,"../data")

    net.train_model(num_epochs,learning_rate)
    
    net.save_model(model_save_path)

    correct,total = net.validate_dataset(net.dataloaders['val'])
    print('Accuracy of the network on the Validation Dataset: %d %%' % (
                100 * correct / total))
