'''
This is an example script to classify multi class classification 
using the Apple sillicon Chip on MNIST dataset using PyTorch.

Library Installation: 
# MPS acceleration is available on MacOS 12.3+
`pip install torch torchvision torchaudio`
'''

# Import Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from time import time


class Model(nn.Module):
    def __init__(self,
                 in_feature=28*28,
                 out_feature=10,
                 hidden_layes=[120, 84]
                 ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_feature, hidden_layes[0])
        self.fc2 = nn.Linear(hidden_layes[0], hidden_layes[1])
        self.fc3 = nn.Linear(hidden_layes[1], out_feature)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        pred = F.log_softmax(x, dim=1)
        return pred


def train(model, train_loader, optimizer, criterion, device, epoch, batch_size):
    train_correct = 0
    train_loss = []
    for batch_num, (X_train, y_train) in enumerate(train_loader, 1):
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_pred = model(X_train.view(batch_size, -1))

        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()

        train_correct += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 200 == 0:
            acc = train_correct.item()*100/(100*batch_num)
            print(
                f'Epoch: {epoch} Batch: {batch_num} Loss: {loss.item()} Accuracy: {acc}')

        train_loss.append(loss.item())

    return np.average(train_loss), train_correct


def test(model, test_loader, criterion, device, epoch, batch_size):
    test_correct = 0
    test_loss = []
    for batch_num, (X_test, y_test) in enumerate(test_loader, 1):
        X_test, y_test = X_test.to(device), y_test.to(device)

        y_pred = model(X_test.view(batch_size, -1))

        loss = criterion(y_pred, y_test)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_test).sum()

        test_correct += batch_corr
        test_loss.append(loss.item())

    print(f"[Epoch: {epoch} Test Loss: {np.average(test_loss)} Correct: {test_correct}  Total: {batch_size*batch_num} Accuracy: {(test_correct/(batch_size*batch_num))*100: .2f}%]")
    print("-"*50)

    return np.average(test_loss), test_correct


def load_trained_model(device):
    # Load saved model
    model = Model().to(device)
    model.load_state_dict(torch.load('ann_mnist.pt'))
    return model


def main():
    # Define Hyperparameters
    epochs = 20
    train_batch_size = 100
    test_batch_size = 500
    transform = transforms.ToTensor()

    # Dataset
    train_dataset = datasets.MNIST(
        root='data/',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='data/',
        train=False,
        download=True,
        transform=transform
    )

    # DataLoader
    torch.manual_seed(101)
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False)

    # Device
    device = 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("Using Device: ", device)

    # Model
    model = Model().to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Trackers
    train_losses = []
    test_loss = []
    train_correct = []
    test_correct = []

    # Tracking Best Model Based on Training Loss
    least_test_loss = np.inf
    # Tracking Time 
    start_time = time()
    # Training and Testing Starts
    for epoch in range(1, epochs+1):
        train_losse, train_corr = train(
            model, train_loader, optimizer, criterion, device, epoch, train_batch_size)
        test_losse, test_corr = test(
            model, test_loader, criterion, device, epoch, test_batch_size)

        # Save Best Model:
        if test_losse < least_test_loss:
            least_test_loss = test_losse
            torch.save(model.state_dict(), 'ann_mnist.pt')
            print("Saved Best Model! \n")
            print("-"*50)

        # Save Trackers
        train_losses.append(train_losse)
        test_loss.append(test_losse)

        train_correct.append(train_corr)
        test_correct.append(test_corr)

    end_time = time()
    print(f"Total Time Taken: {end_time-start_time}")
    # evaluation
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_loss, label='Test/Validation Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
