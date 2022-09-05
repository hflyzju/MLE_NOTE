# -*- coding: utf-8 -*-
import numpy as np
import math

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


y_pred = a + b * x + c * x ** 2 + d * x ** 3
print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')


"""
    绘制sin函数的动画
"""
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
# x = np.arange(0, 6, 0.1)  # 以0.1为单位，生成0到6的数据
# y1 = np.sin(x)
# y2 = np.cos(x)

# 绘制图形
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决标题、坐标轴标签不能是中文的问题
plt.rcParams['axes.unicode_minus'] = False  # 标题等默认是英文输出
plt.plot(x, y, label='sinx')
plt.plot(x, y_pred, linestyle='--', label='sinx_pred')  # 用虚线绘制
plt.xlabel('X')
plt.ylabel('Y')
plt.title('利用pytorch学习拟合sinx图像')
plt.legend()
plt.show()