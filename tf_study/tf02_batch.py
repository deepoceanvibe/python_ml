# 데이터
import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

# 모델링
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

# 컴파일훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=48)

# 평가예측
loss = model.evaluate(x, y)
result = model.predict([6])

print('loss 값 : ', loss)
print('result : ', result)      # batch_size = 2, 6.167427 | 5, 6.0046325 | 7, 5.972841 | 10, 6.052833 | 48, 6.11612