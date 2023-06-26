# 데이터
import numpy as np
x = np.array([1, 2, 3, 5, 4])
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
model.fit(x, y, epochs=1000)

# 평가예측
loss = model.evaluate(x, y)
print('loss : ', loss)
# mse loss :  0.3800000548362732
# mae loss :  0.41345205903053284

result = model.predict([6])
print('6의 예측값 : ', result)
# mse 6의 예측값 :  [[5.7]]
# mae 6의 예측값 :  [[6.0134945]]