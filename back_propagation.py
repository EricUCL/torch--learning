import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x):
    return w * x


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print("\tgrad:", x, y, w.grad.data.item())
        w.data = w.data - 0.01 * w.grad.item()

        w.grad.data.zero_()
    print("progress:", epoch, l.item())
print(w.data.item())
