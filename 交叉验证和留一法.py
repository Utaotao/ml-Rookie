import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
#数据预处理
data_path = r"D:\数据集\iris数据集\iris.data"
names=["sepal length in cm","sepal width in cm","petal length in cm","petal width in cm","class"]
data =pd.read_csv(data_path,sep=",",names=names)
new_data=data.replace(to_replace=["Iris-setosa","Iris-versicolor"],value=[0,1])
new_data=new_data.drop(list(range(100,150)))
X = new_data.iloc[:, :4]
y = new_data.iloc[:, 4]
m, n = X.shape
# # normalization
X = (X - X.mean(0)) / X.std(0)
index = np.arange(m)
#shuffle函数是将列表打乱，生成一个新的列表
np.random.shuffle(index)
X = X.iloc[index]
y = y[index]
# # # 使用sklarn 中自带的api先
# # # k-10 cross validation
lr = linear_model.LogisticRegression(C=0.1)
score = cross_val_score(lr, X, list(y), cv=10)
print(score.mean())
# # 留一法创建一个模型
loo = LeaveOneOut()
accuracy = 0
for train, test in loo.split(X, y):#自动进行留一
    lr_ = linear_model.LogisticRegression(C=2)
    X_train = X.iloc[train,:]
    X_test = X.iloc[test,:]
    y_train = y[train]
    y_test = y[test]
    lr_.fit(X_train, list(y_train))
    accuracy += lr_.score(X_test, list(y_test))
print(accuracy / m)