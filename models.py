import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
import torchvision
from torchvision import models as models
from torchvision import datasets, transforms
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances, euclidean_distances
from skimage.feature import local_binary_pattern

import numpy as np
from numpy import exp, absolute
import matplotlib.pyplot as plt
from PIL import Image
import time, os, copy, math, random, glob, pickle
import cv2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#hyper params
lr = 1e-4
bs = 5
val_split = 0.85
num_epoch = 10
num_classes = 2
labels = ['nom', 'attack']  # class names

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
lbp_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])  # Normalization for grayscale images
])
gray_transform = transforms.Grayscale(num_output_channels=1)


def get_TVT(path, data_transforms):
    dataset = datasets.ImageFolder(path+'train/',transform=data_transforms)
    train_size = math.floor(len(dataset)*val_split)
    val_size = len(dataset) - train_size
    trainset, valset = data.random_split(dataset,lengths=[train_size,val_size])
    testset = datasets.ImageFolder(path+'test/',transform=data_transforms)
    return trainset,valset,testset


class LBPNet(nn.Module):
    def __init__(self, num_classes):
        super(LBPNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def get_model1():
    densenet = torchvision.models.densenet161(pretrained=True)
    # Freeze all the layers except the last few layers
    for param in densenet.parameters():
       param.requires_grad = False
    #densenet.classifier = nn.Sequential(nn.Linear(2208, 512), nn.ReLU(), nn.Linear(512, num_classes), nn.Sigmoid())
    densenet.classifier = nn.Sequential(nn.Linear(2208, num_classes),nn.Sigmoid())
    densenet = densenet.to(device)
    return densenet

def get_model3():
    resnet = torchvision.models.resnet50(pretrained=True)
    # Freeze all the layers except the last few layers
    for param in resnet.parameters():
       param.requires_grad = False
    resnet.fc = nn.Linear(2048,num_classes)
    resnet = resnet.to(device)
    return resnet

def get_model7():
    vgg = torchvision.models.vgg16(pretrained=True)
    # Freeze all the layers except the last few layers
    for param in vgg.parameters():
       param.requires_grad = False
    vgg.classifier[-1] = nn.Sequential(nn.Linear(4096, 512),
                                        nn.Sigmoid(),
                                        nn.Linear(512, num_classes),
                                        nn.Sigmoid())
    vgg = vgg.to(device)
    return vgg

def get_LBP():
    lbp_net = LBPNet(num_classes)  # Create an instance of LBP nn.module
    # Move model to the device
    lbp_net = lbp_net.to(device)
    return lbp_net

#Listing Trained Models
def get_models(m1,m2,m3):
    return [m1,m2,m3]

def train_model(trainset, valset, model, criterion, optimizer, scheduler, num_epochs):
    dataloaders = {
        'train': data.DataLoader(trainset,batch_size=bs,shuffle=True),
        'val' : data.DataLoader(valset,batch_size=bs,shuffle=True)
    }
    dataset_sizes = {'train':len(trainset),'val':len(valset)}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print('bruh')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_acc(model, testset):
    running_corrects = 0
    testloader = data.DataLoader(testset,batch_size=bs,shuffle=True)
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
    return (running_corrects/len(testset))

def PreProcess_img(input):
  img = cv2.imread(input)
  img = cv2.resize(img, (224, 224))
  img = transforms.ToTensor()(img)
  img = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(img)

  # Add batch dimension
  img = img.unsqueeze(0)
  img = img.to(device) #send tensor to GPU
  return img

def get_weighted_score_ft(models,dataset):
    num_models = len(models)
    X = np.empty((0,num_models*num_classes))
    Y = np.empty((0),dtype=int)
    dataloader = data.DataLoader(dataset,batch_size=1,shuffle=True)
    for inputs,labels in dataloader:
        inputs,labels = inputs.to(device),labels.to(device)
        predictions = set()
        with torch.set_grad_enabled(False):
            x = models[0](inputs)
            _, preds = torch.max(x, 1)
            predictions.add(preds)
            for i in range(1,num_models):
                if models[i] is model9:
                  x1 = models[i](gray_transform(inputs))
                else:
                  x1 = models[i](inputs)
                _, preds = torch.max(x1, 1)
                predictions.add(preds)
                x = torch.cat((x,x1),dim=1)
            if len(predictions) > 1:
                X = np.append(X,x.cpu().numpy()*3,axis=0)
            else:
                X = np.append(X,x.cpu().numpy(),axis=0)
            Y = np.append(Y,labels.cpu().numpy(),axis=0)
    return X,Y

def get_weighted_score_img(models,inputs):
  num_models = len(models)
  X = np.empty((0,num_models*num_classes))
  Y = np.empty((0),dtype=int)
  inputs = inputs.to(device)
  predictions = set()
  with torch.set_grad_enabled(False):
      x = models[0](inputs)
      _, preds = torch.max(x, 1)
      predictions.add(preds)
      for i in range(1,num_models):
          if models[i] is model9:
            x1 = models[i](gray_transform(inputs))
          else:
            x1 = models[i](inputs)
            _, preds = torch.max(x1, 1)
          predictions.add(preds)
          x = torch.cat((x,x1),dim=1)
      if len(predictions) > 1:
          X = np.append(X,x.cpu().numpy()*3,axis=0)
      else:
          X = np.append(X,x.cpu().numpy(),axis=0)
      #Y = np.append(Y,labels.cpu().numpy(),axis=0)
  return X

# Display Confidence Score & Predicted Class #
def Cal_Confidence(model,image):
  model.eval()
  #Make a prediction
  with torch.no_grad():
      output = model(image)
  # calculate the softmax of the output
  probabilities = F.softmax(output, dim=1)
  # get the highest probability and its index
  confidence_score, predicted_class = torch.max(probabilities, dim=1)
  print("Confidence score:", confidence_score.item())
  print("Predicted class:", predicted_class.item())
  # Print the predicted class name
  class_index = torch.argmax(probabilities, dim=1).item()
  print('Predicted class:', labels[class_index])

criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
ensemble_accuracy=[]
trainset,valset,testset = get_TVT('/content/drive/MyDrive/DataSets/Palmvein_h/',data_transforms)

# Load the test image

img_path = "/content/drive/MyDrive/DataSets/Palmvein_h/test/probe/nom/041_R_2_4.png"
#img_path = "/content/drive/MyDrive/DataSets/Palmvein_h/test/probe/attack/031_L_2_1.png"
#img_path = "/content/drive/MyDrive/DataSets/Palmvein_h/test/probe/attack/047_L_2_5.png"
#img_path = "/content/drive/MyDrive/DataSets/Palmvein_h/test/probe/attack/047_L_1_3.png"
#img_path = "/content/drive/MyDrive/DataSets/Palmvein_h/test/probe/attack/006_L_1_3.png"
img = PreProcess_img(img_path)

model1 = get_model1()

optimizer = optim.Adam(model1.parameters(), lr=lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.3)
model1 = train_model(trainset, valset, model1, criterion, optimizer, exp_lr_scheduler, num_epoch)

#print(test_acc(model,testset))

torch.save(model1.state_dict(), '/content/drive/MyDrive/DataSets/Save_Models/model1.pth')

model1 = get_model1()
model1.load_state_dict(torch.load('/content/drive/MyDrive/DataSets/Save_Models/model1.pth'))
model1.eval()

Cal_Confidence(model1,img)

print(test_acc(model1,testset))

#avoid
model3 = get_model3()

optimizer = optim.Adam(model3.parameters(),lr=lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.4)
model3 = train_model(trainset, valset, model3, criterion, optimizer, exp_lr_scheduler,num_epoch)

#print(test_acc(model,testset))

#avoid
torch.save(model3.state_dict(), '/content/drive/MyDrive/DataSets/Save_Models/model3.pth')

model3 = get_model3()
model3.load_state_dict(torch.load('/content/drive/MyDrive/DataSets/Save_Models/model3.pth'))
model3.eval()

Cal_Confidence(model3,img)

print(test_acc(model3,testset))

model7 = get_model7()

optimizer = optim.Adam(model7.parameters(), lr=lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.4)
model7 = train_model(trainset, valset, model7, criterion, optimizer, exp_lr_scheduler, num_epoch)

#print(test_acc(model,testset))

torch.save(model7.state_dict(), '/content/drive/MyDrive/DataSets/Save_Models/model7.pth')

model7 = get_model7()
model7.load_state_dict(torch.load('/content/drive/MyDrive/DataSets/Save_Models/model7.pth'))
model7.eval()

Cal_Confidence(model7,img)

print(test_acc(model7,testset))

#For LBP
trainset1,valset1,testset1 = get_TVT('/content/drive/MyDrive/DataSets/Palmvein_h/',lbp_transforms)
model9 = get_LBP()

optimizer = optim.Adam(model9.parameters(), lr=lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.3)
model9 = train_model(trainset1, valset1, model9, criterion, optimizer, exp_lr_scheduler, num_epoch)

torch.save(model9.state_dict(), '/content/drive/MyDrive/DataSets/Save_Models/model9.pth')

model9 = get_LBP()
model9.load_state_dict(torch.load('/content/drive/MyDrive/DataSets/Save_Models/model9.pth'))
model9.eval()

gray_transform = transforms.Grayscale(num_output_channels=1)
gray_img = gray_transform(img)
Cal_Confidence(model9,gray_img)

print(test_acc(model9,testset1))

models= get_models(model1,model3,model9)

#SVM
train_X, train_Y = get_weighted_score_ft(models,trainset)
test_X, test_Y = get_weighted_score_ft(models,testset)

clf = svm.SVC(kernel='poly',break_ties=True).fit(train_X, train_Y)

# Save the SVM model
model_path = '/content/drive/MyDrive/DataSets/Save_Models/svm.pk1'
with open(model_path, 'wb') as file:
    pickle.dump(clf, file)

# Load the SVM model
model_path = '/content/drive/MyDrive/DataSets/Save_Models/svm.pk1'
with open(model_path, 'rb') as file:
    clf = pickle.load(file)

pred = clf.predict(test_X)
acc = accuracy_score(test_Y, pred)
print(acc)  #testing accuracy on whole dataset
clf.score(train_X, train_Y)  #training accuracy on whole dataset

img_X = get_weighted_score_img(models,img)
pred1 = clf.predict(img_X)
print(pred1)

confidence_scores = clf.decision_function(test_X)
print(confidence_scores.max())
print(confidence_scores.min())

def normalisation(score):
  max = 1.0081503554335423
  min = -1370.2244674450371
  return (score - min)/(max - min)

conf1 = clf.decision_function(img_X)
print(conf1)

norm = normalisation(conf1)
print(norm)

