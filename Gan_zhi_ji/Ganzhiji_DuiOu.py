# -*- coding: utf-8 -*- 2

import pandas as pd  # 用pandas读取数据
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
from numpy import arange

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None)  # 读取数据还可以用request这个包

# print(df)  # 输出最后五行数据，看一下Iris数据集格式

"""抽取出前100条样本，这正好是Setosa和Versicolor对应的样本，我们将Versicolor
对应的数据作为类别1，Setosa对应的作为-1。对于特征，我们抽取出sepal length和petal
length两维度特征，然后用散点图对数据进行可视化"""

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
plt.subplot(2,2,1)
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='D', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal lenght')
plt.legend(loc='upper left')
# plt.show()

# train our perceptron model now
# 为了更好地了解感知机训练过程，我们将每一轮的误分类
# 数目可视化出来，检查算法是否收敛和找到分界线
ppn = Perceptron(eta=1, n_iter=5)
ppn.fit(X, y)
print ppn.w_
plt.subplot(2,2,2)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epoches')
plt.ylabel('Number of misclassifications')
plt.legend()
# plt.show()


# 画分界线超平面
def plot_decision_region(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    # markers = ('.', '.', 'o', '^', 'v')
    # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the desicion surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    l1 = arange(x1_min,x1_max,0.1)
    l2 = []
    print ppn.w_[1],ppn.w_[0]
    for i in l1:
        a = -(i*ppn.w_[1]+ppn.w_[0])/ppn.w_[2]
        print i,a
        l2.append(a)
    l3 = np.array(l2)
    plt.subplot(2,1,2)

    # plt.xlim(l1.min()-1, l1.max()+1)
    # plt.ylim(l3.min()-1, l3.max()+1)
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='.')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='.')
    # plt.xlabel('Epoches')
    # plt.ylabel('Number of misclassifications')
    plt.plot(l1,l3,label='chen')
    plt.legend(loc='upper left')
    plt.show()
    #
    # xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    #                        np.arange(x2_min, x2_max, resolution))
    # Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # Z = Z.reshape(xx1.shape)
    # # print Z
    #
    # plt.contour(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # plt.xlim(xx1.min(), xx1.max())
    # plt.ylim(xx2.min(), xx2.max())
    #
    # # plot class samples
    # for idx, cl in enumerate(np.unique(y)):
    #     plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


# plt.subplot(223)

plot_decision_region(X,y,classifier=ppn)
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left',)
# plt.show()