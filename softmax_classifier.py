import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

import torch.nn.functional as F
import torch.optim as optim

y = np.array([1, 0, 0])
z = np.array([0.2, 0.1, -0.1])
print(type(y))
print(type([1, 0, 0]))
y_pred = np.exp(z) / np.exp(z).sum()
loss = (-y * np.log(y_pred)).sum()
print(loss)

y = torch.LongTensor([0])
z = torch.Tensor([[0.2, 0.1, -0.1]])
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z, y)
print(loss.item())
torch.LongTensor([[0]])

y = torch.LongTensor([2, 0, 1])
z1 = torch.Tensor([[0.1, 0.2, 0.9],
                   [1.1, 0.1, 0.2],
                   [0.2, 2.1, 0.1]])
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z1, y)
print(loss.item())
torch.LongTensor([[0]])

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="../dataset/mnist/",
                               train=True,
                               download=True,
                               transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True)
test_dataset = datasets.MNIST(root="../dataset/mnist/",
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Net()

criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader, 0):
        inputs, target = data
        #print(inputs)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index % 300 == 299:
            print('[%d %5d] loss:%.3f' % (epoch + 1, batch_index + 1, running_loss / 300))
            running_loss = 0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
