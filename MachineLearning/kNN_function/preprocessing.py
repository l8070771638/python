import numpy as np

class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self,X):
        """根据训练数据集Ｘ获得数据的均值和方差"""
        assert X.ndim == 2,"The dimension of X must be 2"

        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])

        return self

    def transform(self,X):
        """将Ｘ根据这个StandardScaler进行均值方差归一化处理"""
        assert X.ndim == 2,"这个维度必须为2"
        assert self.mean_ is not None and self.scale_ is not None,\
        "must fit before transform"
        assert X.shape[1] == len(self.mean_),\
        "The feature number of X must be equal to mean_ and std_"
        resX = np.empty(shape=X.shape,dtype=float) #resX 定义为一个空矩阵，shape为传进来的shape
        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col] - self.mean_[col])/self.scale_[col]
        return resX
