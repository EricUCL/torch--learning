import torch.nn
import torch.nn.functional as F

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5000):
    #y_pred = model(x_data)
    optimizer.zero_grad()
    criterion(model(x_data), y_data).backward()
    #print(epoch, loss)


    #loss.backward()
    optimizer.step()


print("w = ", model.linear.weight.data)
print("b = ", model.linear.bias.data)

print(model(torch.Tensor([[0.0]])))
