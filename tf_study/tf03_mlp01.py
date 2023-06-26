import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2.1, 3.1, 4.1, 5.1, 6, 7, 8.1, 9.2, 10.5]])
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
print(x.shape)
print(y.shape)

# 2행 10열 -> 10행 2열로 바꿔서 나열하기
x  = x.transpose()
print(x.shape)

# 모델링
model = Sequential()
model.add(Dense(50, input_dim=2))
model.add(Dense(400))
model.add(Dense(1))

# 컴파일훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=32)

# 결과예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[10, 10.5]])
print('10과 10.5의 예측값 : ', result)