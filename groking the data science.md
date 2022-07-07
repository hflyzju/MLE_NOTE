
## 1 numpy 

|类型|举例|
|--|--|
|numpy列表创建|1. arr1=np.array([[1,2,3],[4,5,6]], dtype='float32') <br> 2. arr1*2, arr1+2, arr1*arr1|
|numpy直接创建|1. np.zeros(100, dtype=int) <br> 2. np.ones((3,3), dtype=float) <br> 3. np.arange(0,20,3) <br> 4. np.linspace(0,1,100) <br> 5.np.random.random((3,3))平均分布 <br> 6. np.random.randint(0, 10, (3,3)) <br> 7.np.random.randnorm(0,1,(3,3))|
|numpy使用|1. arr1[2,3] 3rd row and 4rd column of arr1. <br> 2. arr1[2:4] 切片 <br> 3. arr1[2:3, 2:3] 多维切片 <br>4.arr1[::-1, :] reversing the row <br> 5.arr.copy()|
|reshape和concatenation|1.np.arange(1,10).reshape((3,3)) <br> 2. concate 1d:np.concatenate([x,y,z]) <br> 3. np.vstack(arr1, x)相当于在x轴添加，需要列数相等 <br> 4. np.hstack(arr1, y)相当于在y轴也就是列添加，需要行数相等|
|split|1.np.split(x, (3,6))拆成3行 <br> 2.np.vsplit(x, [2]) 沿着行坐标每2行拆解成一组，列元素不变 <br> 3.np.hsplit(x, [2])沿着列坐标每2列拆成一组，行元素不变|
|函数|1.三角函数 theta = np.linspace(0, 2*np.pi, 4), np.sim(theta), np.cos(theta), np.tan(theta) <br> 2.指数对数Aggregations np.exp(x)=e^x, np.exp2(x)=2^x, np.power(3,x), np.log(x)=ln(x), np.log2(x), np.log10(x)<br> 3.x.mean(), x.max(), x.min(), x.var()|
|比较|1. np.any(arr > 10)是否所有元素大于10 <br> 2. np.all(arr > -1)是否所有元素大于-1 <br> 3. arr[arr > 1] 检索|
