import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBClassifier


# [실습] xgboost, lgbm, catboost 3대장의 성능을 비교해라.

# 성능 비교 결과
# xgboost


# lgbm


# catboost




# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y,
#     train_size=0.7,
#     test_size=0.3,
#     random_state=72,
#     shuffle=True
# )

#scaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

# KFold
kFold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=72
)

# 2. 모델 구성
model = XGBClassifier()

# 2. 훈련, 평가
score = cross_val_score(
    model,
    x, y,
    cv=kFold,
)

print('r2 score : ', score,
      '\n cross_Val_score : ', round(np.mean(score), 4)
      )

