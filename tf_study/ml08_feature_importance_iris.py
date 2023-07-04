import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    random_state=72
)

# scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# kfold
n_splits = 7
random_state = 72
kfold = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)

# 2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
score = cross_val_score(
    model,
    x, y,
    cv=kfold
)

print('acc score : ', score,
      '\n cross_Val_score : ', round(np.mean(score), 4))

############# feature importance 시각화 ###########
# print(model, ": ", model.feature_importances_)
import matplotlib.pyplot as plt

n_features = datasets.data.shape[1]
plt.barh(range(n_features), model.feature_importances_,
         align='center'),
plt.yticks(np.arange(n_features), datasets.feature_names)
plt.title('Cancer Feature Importances')
plt.ylabel('Feature')
plt.xlabel('Importances')
plt.ylim(-1, n_features)
plt.show()