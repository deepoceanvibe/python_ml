import numpy as np
from keras.models import Sequential
from keras.layers import Dense

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

# 모델링
model = Sequential()
model.add(Dense(14, input_dim=1))
model.add(Dense(100))
model.add(Dense(1))


# 컴파일훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

# 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([21])
print('result : ', result)