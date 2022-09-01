## 一、第一部分
https://www.1point3acres.com/bbs/thread-713903-1-1.html


### 1. ML基础概念类
overfitting/underfiting是指的什么
bias/variance trade off 是指的什么
过拟合一般有哪些预防手段
Generative和Discrimitive的区别
Give a set of ground truths and 2 models, how do you be confident that one model is better than another?
#### 1.1 Reguarlization:
L1 vs L2, which one is which and difference
Lasso/Ridge的解释 (prior分别是什么）
Lasso/Ridge的推导
为什么L1比L2稀疏
为什么regularization works
为什么regularization用L1 L2，而不是L3, L4..
#### 1.2 Metric:
precision and recall, trade-off
label 不平衡时用什么metric
分类问题该选用什么metric，and why
confusion matrix
AUC的解释 (the probability of ranking a randomly selected positive sample higher blablabla....)
true positive rate, false positive rate, ROC
Log-loss是什么，什么时候用logloss
还有一些和场景比较相关的问题，比如ranking design的时候用什么metric，推荐的时候用什么.等.不在这个讨论范围内
#### 1.3 Loss与优化
用MSE做loss的Logistic Rregression是convex problem吗
解释并写出MSE的公式, 什么时候用到MSE?
Linear Regression最小二乘法和MLE关系
什么是relative entropy/crossentropy,  以及K-L divergence 他们intuition
Logistic Regression的loss是什么
Logistic Regression的 Loss 推导
SVM的loss是什么
Multiclass Logistic Regression然后问了一个为什么用cross entropy做cost function
Decision Tree split node的时候优化目标是啥
================================
================================
### 2. DL基础概念类
DNN为什么要有bias term, bias term的intuition是什么
什么是Back Propagation
梯度消失和梯度爆炸是什么，怎么解决
神经网络初始化能不能把weights都initialize成0
DNN和Logistic Regression的区别
你为什么觉得DNN的拟合能力比Logistic Regression强
how to do hyperparameter tuning in DL/ random search, grid search
Deep Learning有哪些预防overfitting的办法
什么是Dropout，why it works，dropout的流程是什么 (训练和测试时的区别)
什么是Batch Norm, why it works, BN的流程是什么 (训练和测试时的区别)
common activation functions （sigmoid, tanh, relu, leaky relu） 是什么以及每个的优缺点
为什么需要non-linear activation functions
Different optimizers (SGD, RMSprop, Momentum, Adagrad，Adam) 的区别
Batch 和 SGD的优缺点, Batch size的影响
learning rate过大过小对于模型的影响
Problem of Plateau, saddle point
When transfer learning makes sense.




## 二、第二部分

https://www.1point3acres.com/bbs/thread-714090-1-1.html

这篇将整理一些机器学习模型方面的面经题。同样的，这里只会整理出收集的大厂的好好好几十份面经里出现了的【原题】。大部分题出现的次数都比较高频。
按照我的理解，如果不是面试比较senior的岗位，其中的大部分八股文属于must know。

### 3. ML模型类
#### 3.1 Regression:
Linear Regression的基础假设是什么
what will happen when we have correlated variables, how to solve
explain regression coefficient
what is the relationship between minimizing squared error  and maximizing the likelihood
How could you minimize the inter-correlation between variables with Linear Regression?
if the relationship between y and x is no linear, can linear regression solve that
why use interaction variables
#### 3.2 Clustering and EM:
K-means clustering (explain the algorithm in detail; whether it will converge, 收敛到global or local optimums;  how to stop)
EM算法是什么
GMM是什么，和Kmeans的关系
#### 3.3 Decision Tree
How regression/classification DT split nodes?
How to prevent overfitting in DT?
How to do regularization in DT?
#### 3.4 Ensemble Learning
difference between bagging and boosting
gbdt和random forest 区别，pros and cons
explain gbdt/random forest
will random forest help reduce bias or variance/why random forest can help reduce variance
#### 3.5 Generative Model
和Discrimitive模型比起来，Generative 更容易overfitting还是underfitting
Naïve Bayes的原理，基础假设是什么
LDA/QDA是什么，假设是什么
#### 3.6 Logistic Regression
logistic regression和svm的差别 （我想这个主要是想问两者的loss的不同以及输出的不同，一个是概率输出一个是score）
LR大部分面经集中在logloss和regularization，相关的问题在上个帖子有了这里就不重复了。
#### 3.7 其他模型
Explain SVM, 如何引入非线性
Explain PCA
Explain kernel methods, why to use
what kernels do you know
怎么把SVM的output按照概率输出
Explain KNN
!所有模型的pros and cons （最高频的一个问题）
### 4. 数据处理类
怎么处理imbalanced data
high-dim classification有什么问题，以及如何处理
missing data如何处理
how to do feature selection
how to capture feature interaction
有很多面经题的问题，非常的宽泛，这样的题不在少数。需要有一个比较好的系统的思路去回答。
如果你对其中的问题，以及上个帖子的问题有比较好的见解，欢迎多多回复，帮助多其他复习的人。
如果你对其中的问题，以及上个帖子的问题有哪题不确定，同样欢迎多多回复，大家一起讨论出最合适的答案。
下篇主要归纳一下NLP/RNN的问题，项目经验类的问题，CNN/CV类的问题，实现，推导类的问题。

## 三、第三部分
https://www.1point3acres.com/bbs/thread-714558-1-1.html

Again这是面经整理，不是全面的知识点整理。
面经范围仅限楼主自己扒到的帖子..剔除了一些我觉得过于少见的题目..
欢迎大家把自己的答案贴在回复里..
### 5. implementation 、推导类
写代码实现两层fully connected网络
手写CNN
手写KNN
手写K-means
手写softmax的backpropagation
给一个LSTM network的结构要你计算how many parameters
convolution layer的output size怎么算? 写出公式
       
### 6.项目经验类
训练好的模型在现实中不work,问你可能的原因
Loss趋于Inf或者NaN的可能的原因
生产和开发时候data发生了一些shift应该如何detect和补救
annotation有限的情況下你要怎麼Train model
假设有个model要放production了但是发现online one important feature missing不能重新train model 你怎么办
### 7. NLP/RNN相关
LSTM的公式是什么
why use RNN/LSTM
LSTM比RNN好在哪
limitation of RNN
How to solve gradient vanishing in RNN
What is attention, why attention
Language Model的原理，N-Gram Model
What’s CBOW and skip-gram?
什么是Word2Vec， loss function是什么， negative sampling是什么
### 8. CNN/CV相关
maxpooling， conv layer是什么, 为什么做pooling，为什么用conv lay，什么是equivariant to-translationa, invariant to translation
1x1 filter
什么是skip connection
（楼主没有面任何CV的岗位之前所以基本没收集到什么CV相关的问题）
=========================
### 9. 关于准备考ML 概念的面试的一些建议
1. 如果你简历上提到了一个模型，请确保你对这个模型有着深入全面的了解 （比如很多人可能简历里都提到了XgBoost，但是可能了解并不全面）
举个例子，我简历上提到了Graph Convolutional NN， 我面试的时候就被要求不用包手写一个简单的GCN。
2. 如果job description上提到了某些模型，最好对这些模型也比较熟悉。
3. 对你这个组的domain的相关模型要熟悉。
比如，你面一个明确做NLP的组，那么上述面经就过于基础了。
你或许还要知道 What is BERT， explain the model architecture；what is Transformer model， explain the model architecture；Transformer/BERT 比LSTM好在哪；difference between self attention and traditional attention mechanism；或许你还要知道一些简单的做distill的方法..或许根据组的方向你还要知道ASR, 或者Chat bot等等的方向的一些widely used的模型或者方法。
比如你面一个CTR的组，或许可能你大概至少要稍微了解下wide-and-deep
比如你面一个CV-segment的组，你或许可能要了解DeepMask，U-Net...等等..
你应该不一定需要知道最SOTA的模型，但是知道那些最广为运用的模型或许可能是必要的。这是我的想法，不一定正确。
最后最后 还是希望大家能多多回复一些你们收集到的 高频 的面经。
以及你们的对于某个问题的答案的想法。大家一起讨论。
