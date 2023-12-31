import numpy as np
import pandas as pd

data = pd.DataFrame([
    [2, np.nan, 6, 8, 10],
    [2, 4, np.nan, 8, np.nan],
    [2, 4, 6, 8, 10],
    [np.nan, 4, np.nan, 8, np.nan]
])
print(data)
print(data.shape)   # (4, 5)
data = data.transpose()
print(data.shape)   # (5, 4)

data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

# imputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer

# 1. SimpleImputer : NaN 값을 채우는 방식
# imputer = SimpleImputer()       # 각각의 컬럼값의 평균값으로 채우기
# imputer = SimpleImputer(strategy='mean')     # default와 동일한 뜻
# imputer = SimpleImputer(strategy='median')     # 중간값으로 채우기
# imputer = SimpleImputer(strategy='most_frequent')       # 가장 빈번히 사용되는 값으로
imputer = SimpleImputer(strategy='constant', fill_value=777)       # 특정 값으로

# 2. KNNImputer
# imputer = KNNImputer()      # 평균값으로 채우기
# imputer = KNNImputer(n_neightbors=2)    # 근접한 수 입력

# 3. IterativeImputer
imputer = enable_iterative_imputer()


imputer.fit(data)
data_result = imputer.transform(data)
print(data_result)