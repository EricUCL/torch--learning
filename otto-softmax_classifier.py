import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64


class ProductTrainDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32, skiprows=1)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


train_dataset = ProductTrainDataset(
    "/Users/bytedance/PycharmProjects/torch-learning/train.csv")
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)


#
# class ProductTestDataset(Dataset):
#     def __init__(self, filepath):
#         xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
#         self.len = xy.shape[0]
#         self.x_data = torch.from_numpy(xy[:, :-1])
#         self.y_data = torch.from_numpy(xy[:, [-1]])
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]
#
#
# test_dataset = ProductTestDataset(
#     "/Users/bytedance/PycharmProjects/torch-learning/otto-group-product-classification-challenge/test.csv")
# test_loader = DataLoader(test_dataset,
#                          batch_size=batch_size,
#                          shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(93, 128)
        self.l2 = torch.nn.Linear(128, 64)
        self.l3 = torch.nn.Linear(64, 32)
        self.l4 = torch.nn.Linear(32, 9)

    def forward(self, x):
        # x = x.view(-1, 93)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader, 0):

        inputs, target = Variable(data[0]).float(), Variable(data[1].squeeze()).type(torch.LongTensor)
        # print(inputs)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target - 1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if batch_index % 967 == 966:
        #     print('[%d %5d] loss:%.3f' % (epoch + 1, batch_index + 1, running_loss / 967))
        #     running_loss = 0
        if batch_index % 10 == 0:
            print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data[0]), len(train_loader.dataset),
                loss.item()))
            # print(data[0].shape)
        # print(target.shape)
        # print(epoch, "inputs", data[0], "\n labels", target)


#
# def test():
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, dim=1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
    #  test()
    model.eval()
