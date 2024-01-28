import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import copy 
from sys import exit


#With help from :https://medium.com/@krishna.ramesh.tx/training-a-cnn-to-distinguish-between-mnist-digits-using-pytorch-620f06aa9ffa


batch_size = 64
T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_data = torchvision.datasets.MNIST(root ="", train=True, download=True, transform=T)
val_data = torchvision.datasets.MNIST(root="", train=False, download=True, transform=T)

train_dl = torch.utils.data.DataLoader(train_data, batch_size = batch_size)
val_dl = torch.utils.data.DataLoader(val_data, batch_size = batch_size)


device = "cuda" if torch.cuda.is_available() else "cpu"

def CNNModel():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),

        nn.Conv2d(6, 16, 5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),

        nn.Flatten(),
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    return model

def validate(model, data):
    total = 0
    correct =0 
    for i, (images, labels) in enumerate(data):
        images = images.to(device)
        x = model(images)
        value, pred = torch.max(x,1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct*100./total
    

def training(n_epochs=3, lr=0.001, device="cpu"):
    accuracies = []
    cnn = CNNModel().to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    max_accuracy = 0
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)
            #backpropgation
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()

        accuracy = float(validate(cnn, val_dl))
        accuracies.append(accuracy)
        if (accuracy>max_accuracy):
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving Best Model with Accuracy: ", accuracy)
        print('Epoch:', epoch+1, "Accuracy :", accuracy, '%')
    plt.plot(accuracies)
    #plt.show()
    return best_model 

lenet = training(40, device=device)
print("Done")
torch.save(lenet.state_dict(), "lenet.pth")
lenet = CNNModel().to(device)
lenet.load_state_dict(torch.load("lenet.pth"))
lenet.eval()

def main():
    while(True):
        path = ""
        path = input("Please enter filepath: ")
        if(path == "exit"):
            print("Exiting...")
            exit()
        else:
            img = Image.open(path)
            tensor_img = transforms.ToTensor()
            tensor = tensor_img(img).unsqueeze(0)
            pred = lenet(tensor)
            print("Pytorch Output...")
            print("Done!")
            print(f'Classifier: {pred.argmax()}')
            
main()