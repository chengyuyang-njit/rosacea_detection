# To be finished
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, datasets, models
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

import wandb
wandb.login()

wandb.init(
    project = 'Rosacea Detection',
    config = {
        "learning_rate": 0.001,
        "architecture":"ResNet-18-fine-tune",
        "dataset" :"gen-rosa-norm",
        "epochs" : 50,
    }
)

# Data augmentation and normalization for training
# Just normalization for validation

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(450),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(450),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size = (512,512)),
        transforms.CenterCrop(450),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

data_dir = '../Dataset'
image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir,x),
                                           data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4,
                                              shuffle = True, num_workers = 1)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(class_names)
print("device: ", device)

def train_model(model, criterion, optimizer, scheduler, num_epochs, model_save_name):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    # with TemporaryDirectory() as tempdir:
    ckdir = './checkpoints'
    best_model_params_path = os.path.join(ckdir, model_save_name)

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
        # for phase in ['train']:
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
            

            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                training_loss = running_loss / dataset_sizes[phase]
                training_acc = running_corrects.double() / dataset_sizes[phase]
                wandb.log({"training acc":training_acc, "trainig loss":training_loss})
            else: # phase == 'test'
                val_loss = running_loss / dataset_sizes[phase]
                val_acc = running_corrects.double() / dataset_sizes[phase]
                wandb.log({"validation acc":val_acc, "validation loss":val_loss})

            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    wandb.finish()
    return model

from torchvision.models import ResNet18_Weights
model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr = 0.001, momentum = 0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)


lr = 0.001
momentum = 0.9
gamma = 0.1
model_name = "ResNet-18-fine-tune"
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs = 50, model_save_name=f"{model_name}_lr_{lr}_momentum_{momentum}_gamma_{gamma}.pt")

# On test set
model_ft = models.resnet18()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.load_state_dict(torch.load(
    "./checkpoints/ResNet-18-fine-tune_lr_0.001_momentum_0.9_gamma_0.1.pt", weights_only = True))
model_ft = model_ft.to(device)
model_ft.eval()
running_corrects = 0
TP = 0
TN = 0
FP = 0
FN = 0
for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)
        for i in range(len(preds)):
            if preds[i] == 1 and labels.data[i] == 1:
                TP += 1
            elif preds[i] == 0 and labels.data[i] == 0:
                TN += 1
            elif preds[i] == 1 and labels.data[i] == 0:
                FP += 1
            else: # preds[i] == 0 and labels.data[i] == 1 
                FN += 1

print(TP,TN,FP,FN)
test_accuracy = running_corrects.double() / dataset_sizes['test']
print(f"TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}")
print(f"Test accuray:{test_accuracy}")