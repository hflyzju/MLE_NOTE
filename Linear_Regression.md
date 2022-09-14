### 1. 线性回归梯度求解

```
pre = w1 * x + w0
loss = (pre - y) ** 2
w1_grad = sum(2 * (pre - y) * x) / len(x)
w0_grad = sum(2 * (pre - y) * 1) / len(x)
```

### 2. python实现单变量线性回归

- 教材版本
```python
# https://www.geeksforgeeks.org/univariate-linear-regression-in-python/
class LinearRegressor:
	def __init__(self, x, y, alpha = 0.01, b0 = 0, b1 = 0):
		"""
			x: input feature
			y: result / target
			alpha: learning rate, default is 0.01
			b0, b1: linear regression coefficient.
		"""
		self.i = 0
		self.x = x
		self.y = y
		self.alpha = alpha
		self.b0 = b0
		self.b1 = b1
		if len(x) != len(y):
			raise TypeError("x and y should have same number of rows.")

	def predict(model, x):
		"""Predicts the value of prediction based on
		current value of regression coefficients when input is x"""
		# Y = b0 + b1 * X
		return model.b0 + model.b1 * x

	def cost_derivative(model, i):
		x, y, b0, b1 = model.x, model.y, model.b0, model.b1
		predict = model.predict
		return sum([
			(predict(xi) - yi) * 1
			if i == 0
			else (predict(xi) - yi) * xi
			for xi, yi in zip(x, y)
		]) / len(x)

	def update_coeff(model, i):
		cost_derivative = model.cost_derivative
		if i == 0:
			model.b0 -= model.alpha * cost_derivative(i)
		elif i == 1:
			model.b1 -= model.alpha * cost_derivative(i)

	def stop_iteration(model, max_epochs = 1000):
		model.i += 1
		if model.i == max_epochs:
			return True
		else:
			return False

	def fit(model):
		update_coeff = model.update_coeff
		model.i = 0
		while True:
			if model.stop_iteration():
				break
			else:
				update_coeff(0)
				update_coeff(1)


if __name__ == '__main__':
	linearRegressor = LinearRegressor(
		x =[i for i in range(12)],
		y =[2 * i + 3 for i in range(12)],
		alpha = 0.03
	)
	linearRegressor.fit()
	print(linearRegressor.predict(12))

	# expects 2 * 12 + 3 = 27

```
- output
```
27.000293735134775
```

- 个人版本
```python
# Implementation of gradient descent in linear regression
import numpy as np

x = np.array([2, 6, 5, 7])
y = 2 * x + 1
#
# x = np.array([i for i in range(10)])
# y = np.array([3 * i + 3 for i in range(10)])

a = 1.00
b = 1000.00
best_a = a
best_b = b

pre = a * x + b

iter = 0
lr = 0.03

best_loss = float('inf')
while iter < 100000:
	pre = a * x + b
	loss = (y - pre) ** 2
	if sum(loss) < best_loss:
		best_a = a
		best_b = b
		best_loss = sum(loss)
	a_grad = (0.5 / x.shape[0]) * sum((pre - y) * x) # 为啥这里改为2就不行了, 答：overflow encountered in double_scalars
	b_grad = (0.5 / x.shape[0]) * sum((pre - y))
	# print('a_grad:', a_grad)
	# print('b_grad:', b_grad)
	a -= lr * a_grad
	b -= lr * b_grad
	# print('a:', a)
	# print('b:', b)
	iter += 1


# print('x:', x)
# print('y:', y)

test_x = 12
print('best_a:', best_a)
print('best_b:', best_b)
print(best_a * test_x + best_b)


```
- output
``` 
best_a: 1.999999999999988
best_b: 1.000000000000067
24.999999999999925
```

### 3. 利用pytorch实现线性回归
- 代码
```python
# https://www.geeksforgeeks.org/linear-regression-using-pytorch/

import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]])) # Variable指定可以反向传播
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


class LinearRegressionModel(torch.nn.Module):

	def __init__(self):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(1, 1) # One in and one out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

# our model
our_model = LinearRegressionModel()

criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)

for epoch in range(500):

	# Forward pass: Compute predicted y by passing
	# x to the model
	pred_y = our_model(x_data)

	# Compute and print loss
	loss = criterion(pred_y, y_data)

	# Zero gradients, perform a backward pass,
	# and update the weights.
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print('epoch {}, loss {}'.format(epoch, loss.item()))

new_var = Variable(torch.Tensor([[4.0]]))
pred_y = our_model(new_var)
print("predict (after training)", 4, our_model(new_var).item())

```
- output
```
predict (after training) 4 7.997386932373047
```

### 4. 多变量线性回归(待完成)

```python
# https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/
```