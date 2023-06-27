import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape)  #(20640, 8)
print(y.shape)  #(20640,)
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=1234,
    shuffle=True
)

print(x_train.shape)    # (14447, 8)
print(y_train.shape)    # (14447,)
print(x_test.shape)     # (6193, 8)
print(y_test.shape)     # (6193,)


# 모델구성
model = Sequential()
model.add(Dense(128, input_dim=8))
model.add(128)
model.add(64)
model.add(1)

# 컴파일훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=128)

# 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

## r2 score 결정계수 ##
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)