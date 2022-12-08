# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:42:40 2022

@author: redst
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from customDataset import customData
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform= transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

num_workers = 0

batch_size = 60

epoch_nums = 40

num_classes = 4

dataset = customData(csv_file = './data/TrainingSet/TrainingLabels.csv', root_dir = './data/TrainingSet/resized', transform=transform)
print(dataset.__len__())

trainset = customData(csv_file = './data/TrainingSet/TrainingLabels.csv', root_dir = './data/TrainingSet/resized', transform=transform)
testset = customData(csv_file = './data/TestingSet/TestingLabels.csv', root_dir = './data/TestingSet/resized', transform=transform)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle = True)
testloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle = False)


'''
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
'''


classes = ('cat', 'dog', 'pikachu', 'person', 'drone' )

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
    
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 100 , 6)
        self.fc1 = nn.Linear(100 * 19 * 19, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    
    def trainNet(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


        for epoch in range(epoch_nums):  # loop over the dataset multiple times

            running_loss = 0.0
            loss = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch+1,
                loss
                ))

        print('Finished Training')

    def saveNet(self, path):
        torch.save(self.state_dict(), path)

    def loadNet(self, path):
        self.load_state_dict(torch.load(path))

    def verifyTraining(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 2000 test images: {100 * correct // total} %')

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        
    def getLabels(self, img):
        print("Needs implemented")




save_path = './cifar_net.pth'


'''

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
'''

net = Net()
net.trainNet()
net.saveNet(save_path)

net.loadNet(save_path)
net.verifyTraining()

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(batch_size)))


