import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape)      # (20640, 8)

# drop feature
x = np.delete(x, [3, 4], axis=1)
print(x.shape)      # (20640, 6)

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
model = SVR()

# 2. 훈련, 평가
score = cross_val_score(
    model,
    x, y,
    cv=kFold,
)

print('r2 score : ', score,
      '\n cross_Val_score : ', round(np.mean(score), 4)
      )

# r2 score :  [0.742807   0.72399843 0.73272577 0.74679113 0.72065554] 
#  cross_Val_score :  0.7334


############# feature importance 시각화 ###########
# print(model, ": ", model.feature_importances_)
# import matplotlib.pyplot as plt

# n_features = datasets.data.shape[1]
# plt.barh(range(n_features), model.feature_importances_,
#          align='center'),
# plt.yticks(np.arange(n_features), datasets.feature_names)
# plt.title('Cancer Feature Importances')
# plt.ylabel('Feature')
# plt.xlabel('Importances')
# plt.ylim(-1, n_features)
# plt.show()