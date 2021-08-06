import time
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


iris = load_iris()
print(iris.DESCR)
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)

# 算法参数
params = {
    'booster': 'gbtree',  # 助推器参数，有两个 gbtree：基于树的模型，gblinear：基于线性的模型
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,  # 助推器参数，树的最大深度
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 0,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,  # cpu的线程数
}
# DMatrix 是xgboost 中的数据结构，在使用xgboost时，需要将数据转化为DMatrix的结构，可输入类型有：str，numpy, dataframe, scipy.sparse(稀疏矩阵),dt.Frame
dtrain = xgb.DMatrix(X_train, y_train)
xgb.core

