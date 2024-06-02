import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate
from sklearn import svm, tree
#加载数据
iris = datasets.load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.Series(iris['target_names'][iris['target']])
#线性核
linear_svm = svm.SVC(C=1, kernel='linear')
linear_scores = cross_validate(linear_svm, X, y, cv=5, scoring='accuracy')
print(linear_scores['test_score'].mean())
#高斯核
rbf_svm = svm.SVC(C=1)
rbf_scores = cross_validate(rbf_svm, X, y, cv=5, scoring='accuracy')
print(rbf_scores['test_score'].mean())
#决策树
cart_tree = tree.DecisionTreeClassifier()
tree_scores = cross_validate(cart_tree, X, y, cv=5, scoring='accuracy')
print(tree_scores["test_score"].mean())