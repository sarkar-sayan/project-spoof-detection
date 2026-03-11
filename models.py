import copy
import math
import time
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils import data
from torchvision import datasets, transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters and constants
lr = 1e-4
bs = 5
val_split = 0.85
num_epoch = 10
num_classes = 2
labels = ["nom", "attack"]

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

lbp_transforms = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229]),
    ]
)

gray_transform = transforms.Grayscale(num_output_channels=1)


def get_TVT(
    path: str, transform: transforms.Compose
) -> Tuple[data.Dataset, data.Dataset, data.Dataset]:
    dataset = datasets.ImageFolder(path + "train/", transform=transform)
    train_size = math.floor(len(dataset) * val_split)
    val_size = len(dataset) - train_size
    trainset, valset = data.random_split(dataset, lengths=[train_size, val_size])
    testset = datasets.ImageFolder(path + "test/", transform=transform)
    return trainset, valset, testset


class LBPNet(nn.Module):
    def __init__(self, num_classes: int = num_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def get_model1() -> nn.Module:
    densenet = torchvision.models.densenet161(pretrained=True)
    for param in densenet.parameters():
        param.requires_grad = False
    densenet.classifier = nn.Sequential(nn.Linear(2208, num_classes), nn.Sigmoid())
    densenet = densenet.to(device)
    return densenet


def get_model3() -> nn.Module:
    resnet = torchvision.models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Linear(2048, num_classes)
    resnet = resnet.to(device)
    return resnet


def get_model7() -> nn.Module:
    vgg = torchvision.models.vgg16(pretrained=True)
    for param in vgg.parameters():
        param.requires_grad = False
    vgg.classifier[-1] = nn.Sequential(
        nn.Linear(4096, 512),
        nn.Sigmoid(),
        nn.Linear(512, num_classes),
        nn.Sigmoid(),
    )
    vgg = vgg.to(device)
    return vgg


def get_LBP() -> nn.Module:
    lbp_net = LBPNet(num_classes)
    lbp_net = lbp_net.to(device)
    return lbp_net


def get_models(*models_list: nn.Module) -> List[nn.Module]:
    return list(models_list)


def train_model(
    trainset: data.Dataset,
    valset: data.Dataset,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    num_epochs: int,
) -> nn.Module:
    dataloaders = {
        "train": data.DataLoader(trainset, batch_size=bs, shuffle=True),
        "val": data.DataLoader(valset, batch_size=bs, shuffle=True),
    }
    dataset_sizes = {"train": len(trainset), "val": len(valset)}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    model.load_state_dict(best_model_wts)
    return model


def test_acc(model: nn.Module, testset: data.Dataset) -> torch.Tensor:
    running_corrects = 0
    testloader = data.DataLoader(testset, batch_size=bs, shuffle=True)
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    return running_corrects / len(testset)


def PreProcess_img(input_path: str) -> torch.Tensor:
    img = cv2.imread(input_path)
    img = cv2.resize(img, (224, 224))
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    return img


def get_weighted_score_ft(
    models_list: Sequence[nn.Module], dataset: data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    num_models = len(models_list)
    X = np.empty((0, num_models * num_classes))
    Y = np.empty((0), dtype=int)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = set()
        with torch.set_grad_enabled(False):
            x = models_list[0](inputs)
            _, preds = torch.max(x, 1)
            predictions.add(preds)
            for i in range(1, num_models):
                if models_list[i].__class__.__name__ == "LBPNet":
                    x1 = models_list[i](gray_transform(inputs))
                else:
                    x1 = models_list[i](inputs)
                _, preds = torch.max(x1, 1)
                predictions.add(preds)
                x = torch.cat((x, x1), dim=1)
            if len(predictions) > 1:
                X = np.append(X, x.cpu().numpy() * 3, axis=0)
            else:
                X = np.append(X, x.cpu().numpy(), axis=0)
            Y = np.append(Y, labels.cpu().numpy(), axis=0)
    return X, Y


def get_weighted_score_img(
    models_list: Sequence[nn.Module], inputs: torch.Tensor
) -> np.ndarray:
    num_models = len(models_list)
    X = np.empty((0, num_models * num_classes))
    inputs = inputs.to(device)
    predictions = set()
    with torch.set_grad_enabled(False):
        x = models_list[0](inputs)
        _, preds = torch.max(x, 1)
        predictions.add(preds)
        for i in range(1, num_models):
            if models_list[i].__class__.__name__ == "LBPNet":
                x1 = models_list[i](gray_transform(inputs))
            else:
                x1 = models_list[i](inputs)
                _, preds = torch.max(x1, 1)
            predictions.add(preds)
            x = torch.cat((x, x1), dim=1)
        if len(predictions) > 1:
            X = np.append(X, x.cpu().numpy() * 3, axis=0)
        else:
            X = np.append(X, x.cpu().numpy(), axis=0)
    return X


def Cal_Confidence(model: nn.Module, image: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        output = model(image)
    probabilities = F.softmax(output, dim=1)
    confidence_score, predicted_class = torch.max(probabilities, dim=1)
    print("Confidence score:", confidence_score.item())
    print("Predicted class:", predicted_class.item())
    class_index = torch.argmax(probabilities, dim=1).item()
    print("Predicted class:", labels[class_index])


criterion = nn.CrossEntropyLoss()


def normalisation(score: np.ndarray) -> np.ndarray:
    max_val = 1.0081503554335423
    min_val = -1370.2244674450371
    return (score - min_val) / (max_val - min_val)

