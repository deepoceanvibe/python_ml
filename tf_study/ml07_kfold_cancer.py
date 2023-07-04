import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


# [실습] xgboost, lgbm, catboost 3대장의 성능을 비교해라.

# 성능 비교 결과
# xgboost
# acc score :  [0.93902439 0.93902439 0.96296296 0.98765432 0.98765432 0.97530864
#  1.        ]
#  cross_Val_Score :  0.9702

# lgbm
# acc score :  [0.96341463 0.95121951 0.95061728 1.         0.97530864 0.97530864
#  0.98765432]
#  cross_Val_Score :  0.9719

# catboost
# acc score :  [0.95121951 0.93902439 0.97530864 0.98765432 0.97530864 1.
#  1.        ]
#  cross_Val_Score :  0.9755



# 1. 데이터
datasets = load_breast_cancer()
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
model = CatBoostClassifier()

# 3. 훈련, 평가
score = cross_val_score(
    model,
    x, y,
    cv=kfold
)

print('acc score : ', score,
      '\n cross_Val_Score : ', round(np.mean(score), 4))

# acc score :  [0.97560976 0.97560976 0.96296296 0.98765432 0.98765432 0.97530864 0.96296296]
# cross_Val_Score :  0.9754