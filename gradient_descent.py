x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def loss(x, y):
    return (forward(x) - y) ** 2


def gradient(x, y):
    return 2 * x * (forward(x) - y)


for epoch in range(100):
    for x, y in zip(x_data, y_data):
        print(loss(x, y))
        w -= gradient(x, y) * 0.01

print(w)
