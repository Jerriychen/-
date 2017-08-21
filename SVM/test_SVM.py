#################################################
# SVM: support vector machine
# Author : zouxy
# Date   : 2013-12-12
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import svm as SVM

################## test svm #####################
## step 1: load data
print "step 1: load data..."
dataSet = []
labels = []
fileIn = open('testSet.txt')
for line in fileIn.readlines():
	lineArr = line.strip().split('\t')
	dataSet.append([float(lineArr[0]), float(lineArr[1])])
	labels.append(float(lineArr[2]))

length = len(dataSet)
dataSet = mat(dataSet)
labels = mat(labels).T
mid = int(length*0.8)
train_x = dataSet[0:mid, :]
train_y = labels[0:mid, :]
test_x = dataSet[mid:length, :]
test_y = labels[mid:length, :]

## step 2: training...
print "step 2: training..."
C = 1
toler = 0.001
maxIter = 20
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 0))
# svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter)


## step 3: testing
print "step 3: testing..."
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)

## step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
SVM.showSVM(svmClassifier)