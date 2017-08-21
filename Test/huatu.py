#coding=utf-8
#可以直接运行
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm



#np.lispace(-5,5,500)  生成50个数据，范围[-5，5]
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model
svm.OneClassSVM()
clf = svm.OneClassSVM(nu=0.3, kernel="rbf", gamma=0.3)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
plt.clabel(a,inline=True,fontsize=10)
s = 10
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s)
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s)
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s)
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
plt.show()
















#
# import numpy as np
# import pylab as pl
# import pandas as pd
#
# from sklearn import svm
# from sklearn import linear_model
# from sklearn import tree
# from sklearn.metrics import confusion_matrix
#
# x_min, x_max = 0, 15
# y_min, y_max = 0, 10
# step = .1
# # to plot the boundary, we're going to create a matrix of every possible point
# # then label each point as a wolf or cow using our classifier
# xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
#
# df = pd.DataFrame(data={'x': xx.ravel(), 'y': yy.ravel()})
#
# df['color_gauge'] = (df.x - 7.5) ** 2 + (df.y - 5) ** 2
# df['color'] = df.color_gauge.apply(lambda x: "red" if x <= 15 else "green")
# df['color_as_int'] = df.color.apply(lambda x: 0 if x == "red" else 1)
#
# print "Points on flag:"
# print df.groupby('color').size()
# print
#
# figure = 1
#
# # plot a figure for the entire dataset
# for color in df.color.unique():
# 	idx = df.color == color
# 	pl.subplot(2, 2, figure)
# 	pl.scatter(df[idx].x, df[idx].y, color=color)
# 	pl.title('Actual')
#
# train_idx = df.x < 10
#
# train = df[train_idx]
# test = df[-train_idx]
#
# print "Training Set Size: %d" % len(train)
# print "Test Set Size: %d" % len(test)
#
# # train using the x and y position coordiantes
# cols = ["x", "y"]
#
# clfs = {
# 	"SVM": svm.SVC(degree=0.5),
# 	"Logistic": linear_model.LogisticRegression(),
# 	"Decision Tree": tree.DecisionTreeClassifier()
# }
#
# # racehorse different classifiers and plot the results
# for clf_name, clf in clfs.iteritems():
# 	figure += 1
#
# 	# train the classifier
# 	clf.fit(train[cols], train.color_as_int)
#
# 	# get the predicted values from the test set
# 	test['predicted_color_as_int'] = clf.predict(test[cols])
# 	test['pred_color'] = test.predicted_color_as_int.apply(lambda x: "red" if x == 0 else "green")
#
# 	# create a new subplot on the plot
# 	pl.subplot(2, 2, figure)
# 	# plot each predicted color
# 	for color in test.pred_color.unique():
# 		# plot only rows where pred_color is equal to color
# 		idx = test.pred_color == color
# 		pl.scatter(test[idx].x, test[idx].y, color=color)
#
# 	# plot the training set as well
# 	for color in train.color.unique():
# 		idx = train.color == color
# 		pl.scatter(train[idx].x, train[idx].y, color=color)
#
# 	# add a dotted line to show the boundary between the training and test set
# 	# (everything to the right of the line is in the test set)
# 	# this plots a vertical line
# 	train_line_y = np.linspace(y_min, y_max)  # evenly spaced array from 0 to 10
# 	train_line_x = np.repeat(10, len(train_line_y))  # repeat 10 (threshold for traininset) n times
# 	# add a black, dotted line to the subplot
# 	pl.plot(train_line_x, train_line_y, 'k--', color="black")
#
# 	pl.title(clf_name)
#
# 	print "Confusion Matrix for %s:" % clf_name
# 	print confusion_matrix(test.color, test.pred_color)
# pl.show()



# 可以直接运行
# print(__doc__)
#
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm, datasets
#
# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features. We could
#                       # avoid this ugly slicing by using a two-dim dataset
# y = iris.target
#
# h = .02  # step size in the mesh
#
# # we create an instance of SVM and fit out data. We do not scale our
# # data since we want to plot the support vectors
# C = 1.0  # SVM regularization parameter
# svc = svm.SVC(kernel='linear', C=C).fit(X, y)
# rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
# lin_svc = svm.LinearSVC(C=C).fit(X, y)
#
# # create a mesh to plot in
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
#
# # title for the plots
# titles = ['SVC with linear kernel',
#           'LinearSVC (linear kernel)',
#           'SVC with RBF kernel',
#           'SVC with polynomial (degree 3) kernel']
#
#
# for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     plt.subplot(2, 2, i + 1)
#     plt.subplots_adjust(wspace=0.4, hspace=0.4)
#
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
#
#     # Plot also the training points
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
#     plt.xlabel('Sepal length')
#     plt.ylabel('Sepal width')
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.xticks(())
#     plt.yticks(())
#     plt.title(titles[i])
#
# plt.show()