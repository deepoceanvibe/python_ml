# [실습]
# 1. r2 score를 음수가 아닌 0.5 이하로 만들어라.
# 2. 데이터는 건들지말아라.
# 3. 레이어는 인풋, 아웃풋 포함 7개 이상 만들어라. (히든레이어가 5개 이상이어야 함)
# 4. batch_size=1 이어야 함.
# 5. 히든레이어의 노드(뉴런) 갯수는 10 이상 100 이하로 해라.
# 6. train_size=0.7 이어야 함.
# 7. epochs=100 이상 이어야 함.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x.shape)  # (20,)
print(y.shape)  # (20,)

# train set 70%
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

# test set 30%
x_test = np.array([15, 16, 17, 18, 19, 20])
y_test = np.array([15, 16, 17, 18, 19, 20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    test_size=0.2,
    random_state=64,
    shuffle=True
)

# 모델링
model = Sequential()
model.add(Dense(14, input_dim=1))
model.add(Dense(20))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(20))
model.add(Dense(100))
model.add(Dense(1))


# 컴파일훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,
          y_train,
          epochs=100,
          batch_size=1)

# 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([21])
print('result : ', result)