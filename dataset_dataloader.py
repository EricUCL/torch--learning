import numpy as np
import torch.nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x


dataset = DiabetesDataset("diabetes.csv.gz")
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

model = Model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

losses = []
for epoch in range(800):
    running_loss = 0
    for i, data in enumerate(train_loader, 0):
        # Prepare data
        inputs, labels = data
        # Forward
        loss = criterion(model(inputs), labels)
        print(epoch, i, loss.item())
        running_loss += loss.item()
        if i == 23:
            losses.append(running_loss / 23)
            running_loss = 0
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Update
        optimizer.step()

plt.plot(range(800), losses)
plt.show()
