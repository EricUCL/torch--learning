import torch.nn

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    #y_pred = model(x_data)
    optimizer.zero_grad()
    criterion(model(x_data), y_data).backward()
    #print(epoch, loss)


    #loss.backward()
    optimizer.step()


print("w = ", model.linear.weight.data)
print("b = ", model.linear.bias.data)

print(model(torch.Tensor([[4.0]])))
