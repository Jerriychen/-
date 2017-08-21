#-*-coding:utf-8-*-

import numpy as np


class Perceptron(object):
    """Perceptron classifier.
    Parameters
    ------------
    eta:float,Learning rate (between 0.0 and 1.0)
    n_iter:int,Passes over the training dataset.

    Attributes
    -------------
    w_: 1d-array,Weights after fitting.
    errors_: list,Numebr of misclassifications in every epoch.
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        # Fit training data.先对权重参数初始化，然后对训练集中每一个样本循环，根据感知机算法学习规则对权重进行更新
        # Parameters
        # ------------
        # X: {array-like}, shape=[n_samples, n_features]
        #     Training vectors, where n_samples is the number of samples and n_featuers is the number of features.
        # y: array-like, shape=[n_smaples]
        #     Target values.
        # Returns
        # ----------
        # self: object
        self.w_ = np.zeros(1 + X.shape[1])  # add w_0　　　　　#初始化权重。数据集特征维数+1。
        self.errors_ = []  # 用于记录每一轮中误分类的样本数

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * np.where(target*self.net_input(xi)<=0,target,0)
                # update = self.eta * (target - self.predict(xi))  # 调用了predict()函数
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(np.where(update!=0,1,0))
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]  # 计算向量点乘

    def predict(self, X):  # 预测类别标记
        """return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)