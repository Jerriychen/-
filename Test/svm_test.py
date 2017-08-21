# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/29 20:59
# @Author  : chenzezhong
# @File    : svm_test.py
# @Software: PyCharm Community Edition
# import numpy as np
# from sklearn import svm
# from matplotlib import pyplot as plt
#
# dataSet = []
# labels = []
# fileIn = open('testSet.txt')
# for line in fileIn.readlines():
# 	lineArr = line.strip().split('\t')
# 	dataSet.append([float(lineArr[0]), float(lineArr[1])])
# 	labels.append(float(lineArr[2]))
#
#
# length = len(dataSet)
# print 'length=',length
# # dataSet = np.mat(dataSet)
# # labels = np.mat(labels).T
# mid = int(length*0.8)
# train_x = dataSet[0:mid]
# train_y = labels[0:mid]
# test_x = dataSet[mid:length]
# test_y = labels[mid:length]
#
#
# clf = svm.SVC(C=1,kernel='linear')
# clf.fit(train_x,train_y)
#
# for i in xrange(mid):
# 	if train_x[i] in clf.support_vectors_:
# 		plt.scatter(train_x[i][0], train_x[i][1], color='blue', marker='.')
# 		continue
# 	if train_y[i]==1:
# 		plt.scatter(train_x[i][0],train_x[i][1],color='red', marker='.')
# 	elif train_y[i]==-1:
# 		plt.scatter(train_x[i][0], train_x[i][1], color='yellow', marker='.')
#
# print clf.support_vectors_
# print clf
# count=0
# i=0
#
# length = len(test_x)
# print 'xiao length = ',length
# for i in range(length):
# 	test = np.array(test_x[i]).reshape((1,-1))   #reshape((2,3)) 生成2行3列的数组，
# 	# reshape((1,-1))生成一行数组，reshape((-1,1))生成多行数组，但每行只有一个数
# 	result=clf.predict(test)
# 	if result[0] == test_y[i]:
# 		count += 1
# print 'arcauccy %f'%(float(length)/count)
#
# plt.show()
#




import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt

h = 0.1
x_min, x_max = -1, 1
y_min, y_max = -1, 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
n = xx.shape[0]*xx.shape[1]
x = np.c_[xx.ravel(),yy.ravel()]
# x = np.array([xx.T.reshape(n).T, xx.reshape(n)]).T
y = (x[:,0]*x[:,0] + x[:,1]*x[:,1] > 0.6)
print type(y)
# y = y.reshape(x.shape[0])

train_x = x[:370].tolist()
test_x = x[370:].tolist()
train_y = y[:370].tolist()
test_y = y[370:].tolist()

clf = svm.SVC(C=50,kernel='rbf')
clf.fit(train_x,train_y)

for i in xrange(len(x)):
	if x[i,:].tolist() in clf.support_vectors_.tolist():
		plt.scatter(x[i,0], x[i,1], color='blue', marker='.')
		continue
	if y[i] == True:
		plt.scatter(x[i,0], x[i,1], color='red', marker='.')
	elif y[i] == False:
		plt.scatter(x[i,0], x[i,1], color='yellow', marker='.')

count=0
result = clf.predict(test_x)
for i in range(len(test_y)):
	print result[i], test_y[i]
	if test_y[i]==result[i]:
		count += 1
print 'accurate: %f%%'%(float(count)/len(test_x)*100)

plt.show()