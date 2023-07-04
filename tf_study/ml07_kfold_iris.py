import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
datasets = load_iris
x = datasets.data
y = datasets.target

# scaler
scaler = RobustScaler()
x = scaler.fit_transform(x)

# kfold
n_splits = 7
random_state = 72
kfold = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)

# 2. 모델
model = RandomForestClassifier

# 3. 훈련, 평가
score = cross_val_score(
    model,
    x, y,
    cv=kfold
)

print('acc score : ', score,
      '\n cross_Val_score : ', round(np.mean(score), 4))