from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
regensburg_pediatric_appendicitis = fetch_ucirepo(id=938)
pd.set_option('display.max_columns', None)
# data (as pandas dataframes)
X = regensburg_pediatric_appendicitis.data.features
y = regensburg_pediatric_appendicitis.data.targets
y=y["Severity"]
X=X.replace(to_replace=["female","male"],value=[0,1])
X=X.drop(X.columns[6:53],axis=1)
# X=X.astype(float)
y=y.replace(["uncomplicated","complicated"],[0,1])
result = pd.concat([X, y], axis=1)
result=result.dropna(how="any",axis=0)
result=result.reset_index(drop=True)
X=result.iloc[:,0:-2]
y=result.iloc[:,-1]
m,n= X.shape
# normalization
X = (X - X.mean(0)) / X.std(0)
# #
index = np.arange(m)
# #shuffle函数是将列表打乱，生成一个新的列表
np.random.shuffle(index)
X = X.iloc[index]
y = y[index]
# # # 使用sklarn 中自带的api先
# # # k-10 cross validation
lr = linear_model.LogisticRegression(C=2)
score = cross_val_score(lr, X, list(y), cv=10)
print(score.mean())
# # # # 留一法创建一个模型
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
