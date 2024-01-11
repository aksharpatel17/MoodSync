import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import numpy as np

# Hyperparameters
num_epochs = 10
num_classes = 4
learning_rate = 0.001

# Image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset path
local_dataset_path = "C:\\Users\\pc\\Downloads\\captured_frames\\dataset"
dataset = ImageFolder(root=local_dataset_path, transform=transform)

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

# Class labels
classes = ('anger', 'focused', 'neutral', 'tired')

# Extract labels for train, validation, and test sets
y_train = np.array([y for x, y in iter(train_dataset)])
print(len(y_train))
y_val = np.array([y for x, y in iter(val_dataset)])
print(len(y_val))
y_test = np.array([y for x, y in iter(test_dataset)])
print(len(y_test))

# Convolutional Neural Network (CNN) model definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        out_size = lambda i, k, p, s: (i - k + 2 * p) // s + 1
        out1 = out_size(96, 3, 1, 1)  # Output size after conv1
        out2 = out_size(out1, 2, 0, 2)  # Output size after pool1
        out3 = out_size(out2, 3, 1, 1)  # Output size after conv2
        out4 = out_size(out3, 2, 0, 2)  # Output size after pool2
        expected_input_size = out4 * out4 * 64  # Assuming 64 channels in the last conv layer

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(expected_input_size, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        # Forward pass
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

if __name__ == '__main__':
    # Initialize CNN model
    model = CNN()

    # Define NeuralNetClassifier
    net = NeuralNetClassifier(
        model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        max_epochs=num_epochs,
        lr=learning_rate,
        batch_size=32,
        iterator_train__shuffle=True,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    # Training loop with validation
    for epoch in range(num_epochs):
        # Training
        net.fit(train_dataset, y=y_train)

        # Validation
        val_acc = net.score(val_dataset, y=y_val)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_acc * 100:.2f}%')


    # Save the trained model
    torch.save(net.module_, 'main_model.pth')

    # Testing
    test_acc = net.score(test_dataset, y_test)
    print(f'Test Accuracy: {test_acc * 100:.2f}%')

    # Evaluation metrics
    y_pred = net.predict(test_dataset)
    accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # Precision, Recall, and F1-measure
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-measure: {f1:.2f}')