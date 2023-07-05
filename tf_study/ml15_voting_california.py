import numpy as np
from sklearn.datasets import _california_housing
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. 데이터
datasets = _california_housing()
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
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#kfold
n_splits=5,
kfold = StratifiedKFold(
    n_splits=n_splits,
    random_state=62,
    shuffle=True
)

# 2. 모델
nc_model = KNeighborsClassifier(n_neighbors=7)
rf_model = RandomForestClassifier()
dt_model = DecisionTreeClassifier()
cat = CatBoostClassifier
lgbm = LGBMClassifier
xgb = XGBClassifier

model = VotingClassifier(
    # estimators=[('nc_model', nc_model),
    #             ('rf_model', rf_model),
    #             ('dt_model',dt_model)],
    estimators=[('cat', cat), ('lgbm', lgbm),
                ('xgb', xgb)],
                voting='soft',
                n_jobs=-1
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
from sklearn.metrics import accuracy_score

# classifiers = [rf_model, dt_model, nc_model]
classifiers = [cat, lgbm, xgb]
for model in classifiers:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4f}'.format(class_names, score))

    result = model.score(x_test, y_test)
    print(result)