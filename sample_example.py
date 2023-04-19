"""
This is a simple test script to write deep learning model to classify cats vs dogs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50






# Check if mps availabe on apple sillicon, otherwise use cpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_model():
    '''
    This function returns a pretrained resnet50 model.
    '''
    model = resnet50(pretrained=True)
    # freeze all layers except last two layers
    for param in model.parameters()[:-2]:
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    return model


def train_test_dataset():
    train_data = datasets.ImageFolder(
        root='./media/train',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        )
    )

    test_data = datasets.ImageFolder(
        root='./media/test',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        )
    )
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs=5):
    '''
    This function trains the model and returns the model with the best accuracy.
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0            
    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            


def main():
    model = get_model()
    train_loader, test_loader = train_test_dataset()
    train_model(model, train_loader, test_loader)
    print('Done')


if __name__ == '__main__':
    main()