import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-1.0, 3.1, 0.1):
        print("w = ", w)
        print("b = ", b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / len(x_data))
        print("MSE=", l_sum / len(x_data))
print(w_list)
print(mse_list)

#
# nx, ny = (3, 2)
# x = np.linspace(0, 10, 11)
# y = np.linspace(0, 100, 101)
# xv, yv = np.meshgrid([1,2,3,4,5], [1,2,3])
# print(xv)
# print(yv)
# print(xv[1,1])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X, Y = np.meshgrid(w_list, b_list)
ax.scatter3D(w_list, b_list, np.array(mse_list))
plt.show()
