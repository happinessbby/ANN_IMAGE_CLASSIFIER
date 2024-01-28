import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms



# Courtesy of https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

batch_size = 64
T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(root="", train=True, transform=T)
val_data = torchvision.datasets.MNIST(root="", train=False, transform=T)

train_dl = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_dl = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = False)


device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 40
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer)
    test(val_dl, model)
print("Done!")

torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
model.eval()

def main():
    while(True):
        path = ""
        path = input("Please enter filepath: ")
        if(path == "exit"):
            print("Exiting...")
            exit()
        else:
            img = Image.open(path)
            transform = transforms.ToTensor()
            tensor = transform(img).unsqueeze(0)
            pred = model(tensor)
            print("Pytorch Output...")
            print("Done!")
            print(f'Classifier: {pred.argmax()}')
                  
main()
