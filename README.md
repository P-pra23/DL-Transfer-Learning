# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement 

The problem statement for this experiment is to develop an image classification model that can accurately distinguish between 'defect' and 'notdefect' semiconductor chip images. This is a binary classification task, where the goal is to leverage transfer learning using a pre-trained VGG19 model to effectively classify new, unseen chip images.


## Neural Network Model

<img width="1043" height="802" alt="image" src="https://github.com/user-attachments/assets/e89b0214-396e-402c-9891-22c433677482" />


## DESIGN STEPS
### STEP 1: 
Import required libraries and define image transforms.

### STEP 2: 
Load training and testing datasets using ImageFolder.
### STEP 3: 
Visualize sample images from the dataset.
### STEP 4: 
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.
### STEP 5: 
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.
### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.


## PROGRAM

### Name: Prathikshaa

### Register Number: 2122241000


```PYTHON
from google.colab import drive
drive.mount('/content/drive')
     

import torch
import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchsummary import summary
     

## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])
     

!unzip -qq /content/drive/MyDrive/chip_data.zip -d data
     
replace data/dataset/test/defect/D2_C97.jpg? [y]es, [n]o, [A]ll, [N]one, [r]ename: 

# Load dataset from a folder (structured as: dataset/class_name/images)
dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
     

# Display some input images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()
     

# Show sample images from the training dataset
show_sample_images(train_dataset)
# Get the total number of samples in the training dataset
print(f"Total number of training samples: {len(train_dataset)}")

# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")
     
Total number of training samples: 172
Shape of the first image: torch.Size([3, 224, 224])

# Get the total number of samples in the testing dataset
print(f"Total number of test samples: {len(test_dataset)}")

# Get the shape of the first image in the dataset
first_image1, label = test_dataset[0]
print(f"Shape of the first image: {first_image1.shape}")
     
Total number of test samples: 121
Shape of the first image: torch.Size([3, 224, 224])

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

     

model=models.vgg19(weights=VGG19_Weights.DEFAULT)
device=t.device("cuda" if t.cuda.is_available() else "cpu")
     

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
     

from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)
     

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
     

summary(model, input_size=(3, 224, 224))

# Freeze all layers except the final layer
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers
     

criterion=nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
     

def train_model(model,train_loader,test_loader,num_epochs=10):
  train_losses=[]
  val_losses=[]
  for epoch in range(num_epochs):
    running_loss=0.0
    for images,labels in train_loader:
      images,labels=images.to(device),labels.to(device)
      optimizer.zero_grad()
      outputs=model(images)
      loss=criterion(outputs,labels.unsqueeze(1).float())
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
    train_losses.append(running_loss/len(train_loader))

    model.eval()
    val_loss=0.0
    with t.no_grad():
      for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels.unsqueeze(1).float())
        val_loss+=loss.item()
    val_losses.append(val_loss/len(test_loader))
    model.train()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")
  plt.figure(figsize=(8,6))
  plt.plot(range(1,num_epochs+1),train_losses,label="Train Loss",marker="o")
  plt.plot(range(1,num_epochs+1),val_losses,label="Validation Loss",marker="s")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.title("Training and validation Loss")
  plt.legend()
  plt.show()
     

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
     

# Train the model
# Write your code here


train_model(model,train_loader,test_loader)

## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with t.no_grad():
        for images, labels in test_loader:
          images = images.to(device)
          labels = labels.float().unsqueeze(1).to(device)

          outputs = model(images)

          probs = t.sigmoid(outputs)

          predicted = (probs > 0.5).int().squeeze()

          total += labels.size(0)

          correct += (predicted == labels.squeeze().int()).sum().item()

          all_preds.extend(predicted.cpu().numpy())

          all_labels.extend(labels.squeeze().cpu().numpy().astype(int))

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Name: Prathikshaa  ")
    print("Register Number:  212224100043 ")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Name: Prathikshaa  ")
    print("Register Number: 212224100043 ")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

     

# Evaluate the model
# write your code here
test_model(model, test_loader)

def predict_image(model,image_index,dataset):
    model.eval()
    image,label=dataset[image_index]

    with t.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

        prob = t.sigmoid(output)
        predicted = (prob > 0.5).int().item()

    class_names = dataset.classes

    image_to_display = transforms.ToPILImage()(image)

    plt.figure(figsize=(4,4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis('off')
    plt.show()
     

# Example Prediction
predict_image(model, image_index=55, dataset=test_dataset)

#Example Prediction
predict_image(model, image_index=25, dataset=test_dataset)
```


### OUTPUT

<img width="707" height="752" alt="image" src="https://github.com/user-attachments/assets/09eb8892-20f6-4aed-8f22-2ed1310d318f" />
<img width="692" height="771" alt="image" src="https://github.com/user-attachments/assets/4bbf6414-c226-456e-95c6-7a4acc0f6868" />

## Training Loss, Validation Loss Vs Iteration Plot

<img width="902" height="682" alt="image" src="https://github.com/user-attachments/assets/ff5c3c4c-462a-4759-b9ca-0101cc17e6c4" />


## Confusion Matrix

<img width="876" height="732" alt="image" src="https://github.com/user-attachments/assets/e17990ee-b798-4035-832b-843c70dcd9f7" />


## Classification Report

<img width="585" height="267" alt="image" src="https://github.com/user-attachments/assets/ae53b785-1704-4763-ad68-86281d7ceaf2" />


### New Sample Data Prediction

<img width="466" height="492" alt="image" src="https://github.com/user-attachments/assets/ab30cd00-2fc0-4b43-a179-97a2d3219ce8" />
<img width="432" height="482" alt="image" src="https://github.com/user-attachments/assets/cbe1616b-c4cc-45e0-ac11-6d71cbae86e9" />




## RESULT

Thus the python program to develop an image classification model using transfer learning with VGG19 architecture is executed successfully.
