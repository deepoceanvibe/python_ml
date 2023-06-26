# 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))    # 입력층
model.add(Dense(4))    # 히든레이어1
model.add(Dense(7))    # 히든레이어2
model.add(Dense(4))    # 히든레이어3
model.add(Dense(2))    # 히든레이어4
model.add(Dense(1))    # 출력층

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')     # mse : 평균 제곱 오차 , mae = 평균 절대값 오차 
model.fit(x, y, epochs=100)

# 로스평가, 예측
loss = model.evaluate(x, y)
print('loss 값 : ', loss)
result = model.predict([4])
print('result : ', result)      # result :  [[4.051786]] loss 값 :  0.0005207248032093048
