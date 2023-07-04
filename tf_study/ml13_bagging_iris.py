from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np

# 1. 데이터
datasets = load_iris()
x = datasets['data']
y =  datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

# scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# kfold
n_splts = 5
random_state = 62
kfold = StratifiedKFold(
    n_splits=n_splts,
    random_state=random_state,
    shuffle=True
)

# 2. 모델
dt_model = DecisionTreeClassifier()
model = BaggingClassifier(
    dt_model,
    n_estimators=100,
    n_jobs=1,
    random_state=random_state
)

# 3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

# 4. 평가
result = model.score(x_test, y_test)

score = cross_val_score(
    model,
    x, y,
    cv=kfold
)

print('acc score : ', score,
      '\n cross_Val_score : ', round(np.mean(score), 4))

# acc score :  [1.         0.9        0.83333333 0.96666667 0.96666667] 
#  cross_Val_score :  0.9333