import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 1, 1, 2, 1.1, 1.2, 1.4, 1.5, 1.6],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

x = x.transpose()


# 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(500))
model.add(Dense(200))
model.add(Dense(1))

# 컴파일훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=32)


# 예측 [[10, 1.6, 1]]
loss = model.evaluate(x, y)
result = model.predict([[10, 1.6, 1]])

print('loss 값 : ', loss)       # 1.4551915445207286e-12 7.275957722603643e-13
print('예측 값 : ', result)     #  [[19.999998]] [[20.000004]]
