import pandas as pd
# reading csv files\
names=["sepal length in cm","sepal width in cm","petal length in cm","petal width in cm","class"]
data =  pd.read_csv(r"D:\数据集\iris数据集\iris.data", sep=",",names=names)
print(data)
