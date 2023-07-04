import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import time

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

# scaler
scaler = MinMaxScaler()
x_trian = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# kfold
n_splits = 5
random_state = 62
kfold = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)

param = [
    {'n_estimators' : [100, 500], 'max_depth':[6, 8, 10, 12], 'n_jobs' : [-1, 2, 4]},  
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]}, 
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]


# 2. 모델
model = RandomForestClassifier(max_depth=6, n_jobs=4)


# 3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time


print('model_score : ', model.score(x_test, y_test))
print('걸린시간 : ', end_time)

# 최적의 파라미터 :  {'max_depth': 6, 'n_estimators': 100, 'n_jobs': 4}
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, n_jobs=4)
# best_score :  0.9714285714285713
# model_score :  0.37777777777777777
# 걸린시간 :  19.85700798034668

# 개선이후
# model_score :  0.37777777777777777
# 걸린시간 :  0.15851831436157227