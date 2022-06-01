Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization
https://aman.ai/coursera-dl/improving-deep-neural-networks/


- Train / Dev / Test Sets
Make sure the dev and test set are coming from the same distribution.

- Bias / Variance
If your model is underfitting (logistic regression of non linear data) it has a “high bias”
If your model is overfitting then it has a “high variance”
Your model will be alright if you balance the Bias / Variance

- Basic Recipe for Machine Learning
  - high bias:    

Try to make your NN bigger (size of hidden units, number of layers)
Try a different model that is suitable for your data.
Try to run it longer.
Different (advanced) optimization algorithms.
  - high variance:  

More data.
Try regularization.
Try a different model that is suit


- Regularization

Adding regularization to NN will help it reduce variance (overfitting)
L1 matrix norm:sum of absolute values of all w
L2 matrix norm because of arcane technical math reasons is called Frobenius norm:sum of all w squared

The L1 regularization version makes a lot of w values become zeros, which makes the model size smaller.
L2 regularization is being used much more often.