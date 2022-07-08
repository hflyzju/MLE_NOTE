## 1. 总结
|类型|样例|说明|
|-|-|-|
|创建|torch.tensor([[1,2],[3,4]]) <br> torch.from_numpy(np.array) <br> torch.ones_like(x_data) <br> torch.rand_like(x_data, dtype=torch.float) <br> torch.rand((2,3,))：随机均匀分布<br> torch.randn(1,1,32,32):标准正态分布 <br> torch.ones((2,3,)) <br> torch.zeros((2,3,))|torch.cuda.is_available <br> torch.rand(2,3)|
|indexing and slicing|tensor[:,1] = 0:将所有行的第一列转化为1 <br> torch.cat([tensor, tensor, tensor], dim=0) ：2维的向量，则是在$行$上面进行拼接。<br> torch.cat([tensor, tensor, tensor], dim=1) ：2维的向量，则是在$列$上面进行拼接。|
|乘法|torch.mul(tensor1, tensor2):对应的元素相乘，不是点乘，与tensor1 * tensor2 以及tensor1.mul(tensor2)等价|
|矩阵乘法|torch.matmul(tensor1, tensor2)等价于tensor1.matmul(tensor2)|tensor2.T转置|
|in-place|tensor1.add_(5):有个下划线，代表in-place加|x.copy_(y)|
|numpy|torch.ones(5).numpy() <br> torch.from_numpy(n)|注意torch.from_numpy(arr)当arr发生改变的时候，生产的tensor也会改变|
|梯度|import torch<br> a = torch.tensor([2., 3.], requires_grad=True)<br> b = torch.tensor([6., 4.], requires_grad=True)<br> Q = 3*a**3 - b**2<br> external_grad = torch.tensor([1., 1.])<br> Q.backward(gradient=external_grad)<br> # check if collected gradients are correct<br> print(9*a**2 == a.grad)<br> print(-2*b == b.grad)<br> print(a.grad)<br>|tensor([True, True])<br> tensor([True, True])<br> tensor([36., 81.])<br>|
|模型构建|nn.Conv2d(1, 6, 5):5*5的卷积核，1进来，6出去<br> nn.Linear(dim_in, dim_out)<br>F.max_pool2d(F.relu(self.conv1(x)), (2, 2)):(2,2)为池化层的大小|import torch<br> import torch.nn as nn<br> import torch.nn.functional as F<br> |
|优化器|import torch.optim as optim<br> # create your optimizer<br> optimizer = optim.SGD(net.parameters(), lr=0.01)<br> # in your training loop: <br>optimizer.zero_grad()   # zero the gradient buffers <br>output = net(input) <br>loss = criterion(output, target) <br>loss.backward() <br>optimizer.step()    # Does the update|import torch.optim as optim|
|变换|torch.flatten(x, 1) # flatten all dimensions except the batch dimension,将除了batch的都拉平 <br> tensor.vies(1, -1):将所有的都拉平，变成1*all_dim的维度 <br> tensor.view(8,1,-1).shape: 8 * 1 * all_dims <br> tensor.transpose(3,2).shape 8 * 1 * 32 * 64 -> 8 * 1 * 64 * 32 <br> scores.masked_fill_(attn_mask, -1e9):填充||
|Embedding|nn.Embedding(dim_in, dim_out):相当于把one hot转化成为向量||
|变换|pos.unsqueeze(0)：添加一维度<br> pos.unsqueeze(0).expand_as(x):在添加一个维度的基础上，再拓展batch次，从1 *768 => batch * 768||
|GPU|device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')<br> # Assuming that we are on a CUDA machine, this should print a CUDA device: <br>print(device)<br> net.to(device) <br> inputs, labels = data[0].to(device), data[1].to(device)||
|BERT相关|l1(embed).view(8, -1, 12, 64).transpose(1,2).shape # Batch_size * seq_len * model_dim => Batch_size * n_heads * seq_len * d_k, (8, 30, 768) => (8, 12, 30, 64)||

## 2. 构建网络


### 2.1 基础构件

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print(x.shape)
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = self.fc3(x)
        print(x.shape)
        return x


net = Net()
print(net)
"""
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
"""
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
"""
10
torch.Size([6, 1, 5, 5])
"""

input = torch.randn(1,1,32,32)

# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

# loss函数
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

"""
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([ 0.0254,  0.0225, -0.0100, -0.0329, -0.0150,  0.0164])
"""
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

- 使用优化器更新loss

```python
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```

- 不适用优化器版本

```python
# loss函数
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
net.zero_grad()     # zeroes the gradient buffers of all parameters
loss.backward()
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

- 完整版本

```python
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

input = torch.randn(1, 1, 32, 32)

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```



### 2.2 图像分类网络训练与测试
```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

```

## 3. BERT模型


```python
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seg_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.token_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

def get_attn_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # batch_size * 1 * len_k
    return pad_attn_mask.expand(batch_size, len_q, len_k) # batch_size * len_q * len_k

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v)
        return context, attn

class MultiheadAttention(nn.Module):

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.wq = nn.Linear(d_model, d_k * n_heads)
        self.wk = nn.Linear(d_model, d_k * n_heads)
        self.wv = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)

        # l1(embed).view(8, -1, 12, 64).transpose(1,2).shape # Batch_size * seq_len * model_dim => Batch_size * n_heads * seq_len * d_k, (8, 30, 768) => (8, 12, 30, 64)
        qs = self.wq(Q).view(batch_size, -1, n_heads, d_q).transpose(1,2)
        kw = self.wk(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        vs = self.wv(Q).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        # (8, 12, 30, 64) matmul (8, 12, 64, 30) => 8, 12, 30, 30
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # (8, 12, 30, 30) matmul (8, 12, 30, 64) => 8, 12, 30, 64
        context = torch.matmul(attn, v)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn

```

## 4. pytorch拟合函数

- 拟合sin函数

```python
dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(4000):
    y_pred = a * x + b * (x ** 3) + c * (x ** 5) + d * (x ** 7)

    loss = (y_pred - y).pow(2).mean()
    if t % 100 == 99:
        print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} x + {b.item()} x^3 + {c.item()} x^5 + {d.item()} x^7')
print('1/1! = {}'.format(1./1))
print('1/3! = {}'.format(1./(1*2*3)))
print('1/5! = {}'.format(1./(1*2*3*4*5)))
print('1/7! = {}'.format(1./(1*2*3*4*5*6*7)))


```

- 回归

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

for t in range(200):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()


```