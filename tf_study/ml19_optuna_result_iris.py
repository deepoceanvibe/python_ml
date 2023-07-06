import numpy as np
from catboost import CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# scaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# kfold
n_splits = 7
random_state = 865
kfold = StratifiedKFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)

# 2. 모델
model = CatBoostClassifier(
    n_estimators = 839,
    depth = 14,
    fold_permutation_block = 126,
    learning_rate = 0.5550025342132437,
    od_pval = 0.2041630181764047, 
    l2_leaf_reg = 0.37486589313190555,
    random_state = 865
)

# 3. 훈련, 평가 
score = cross_val_score(
    model,
    x, y,
    cv = kfold
)

print('acc score : ', score,
      '\n cross_Val_Score : ', round(np.mean(score), 4))

# acc score :  [1.         0.90909091 0.95454545 0.9047619  0.9047619  1.
#  1.        ] 
#  cross_Val_Score :  0.9533

# acc score :  [0.90909091 0.95454545 0.90909091 1.         1.         0.95238095
#  0.9047619 ]
#  cross_Val_Score :  0.9471