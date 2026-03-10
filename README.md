# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Image classification is a fundamental problem in computer vision, where the goal is to assign an input image to one of the predefined categories. Traditional machine learning models rely heavily on handcrafted features, whereas Convolutional Neural Networks (CNNs) automatically learn spatial features directly from pixel data.

In this experiment, the task is to build a Convolutional Deep Neural Network (CNN) to classify images from the FashionMNIST dataset into their respective categories. The trained model will then be tested on new/unseen images to verify its effectiveness.
## Neural Network Model

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/b6dfa154-2064-4760-8b8c-8df202567d42" />

## DESIGN STEPS

## STEP 1: Problem Statement
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

### STEP 2:Dataset Collection
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

## STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

## STEP 4:Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

## STEP 5:Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

## STEP 6:Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

## STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.


## PROGRAM

### Name: KUKKADAPU CHARAN TEJ
### Register Number: 212224040167
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
      x=self.pool(torch.relu(self.conv1(x)))
      x=self.pool(torch.relu(self.conv2(x)))
      x=self.pool(torch.relu(self.conv3(x)))
      x=x.view(x.size(0), -1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x  
```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: KUKKADAPU CHARAN TEJ')
        print('Register Number: 212224040167')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


```

## OUTPUT
### Training Loss per Epoch

<img width="553" height="270" alt="image" src="https://github.com/user-attachments/assets/7ce7e65c-cd37-428d-86b9-a1d45b4fa84d" />

### Confusion Matrix

<img width="658" height="547" alt="image" src="https://github.com/user-attachments/assets/918824a7-763f-46f5-ab54-23428e9e9e5e" />


### Classification Report

<img width="719" height="483" alt="image" src="https://github.com/user-attachments/assets/05203c7a-2a02-43b4-aa39-047e1ccd85d6" />



### New Sample Data Prediction

<img width="1241" height="518" alt="image" src="https://github.com/user-attachments/assets/95d831cb-161f-4eea-a53f-280bdfe93788" />


## RESULT
A CNN was successfully implemented for handwritten digit classification using MNIST. The model achieved high accuracy (~98%) and correctly classified new unseen digit samples.
