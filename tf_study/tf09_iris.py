import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import time

# 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (150, 4) (150,)
print(datasets.feature_names)       # 'sepal length (cm)', 'sepal width (cm)'
print(datasets.DESCR)       

#### 원핫인코딩 ####
from keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)      # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

print(x_train.shape, x_test.shape)      # (105, 4) (45, 4) -> 4는 인풋
print(y_train.shape, y_test.shape)      # (105, 3) (45, 3) -> 3은 아웃풋


# 모델구성
model = Sequential()
model.add(Dense(32, input_dim=4))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation='softmax'))


# 컴파일훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['mse', 'accuracy'])
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=32)
end_time = time.time() - start_time
print('걸린시간 : ', end_time)

# 평가예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mse : ', mse)
print('accuracy : ', accuracy)

# loss :  0.016855524852871895
# mse :  0.0018411976052448153
# accuracy :  1.0


# 이 스크립트는 Keras를 사용하여 붓꽃(iris) 데이터셋을 분류하는 신경망 모델을 구성하고 훈련하는 코드입니다.
# 데이터셋은 사이킷런의 load_iris 함수를 통해 로드되며,
# 총 150개의 샘플과 4개의 특성으로 구성됩니다.

# 원핫인코딩을 위해 keras.utils.to_categorical 함수를 사용하여 레이블 데이터 y를 변환합니다.
# 이렇게 하면 3개의 클래스에 대한 원핫인코딩 벡터로 변환됩니다.

# 데이터는 학습 세트와 테스트 세트로 분할되며,
# 학습 세트는 105개의 샘플을 가지고 있고,
# 테스트 세트는 45개의 샘플을 가지고 있습니다.

# 모델은 Sequential 모델로 구성되어 있으며, 총 4개의 Dense(완전 연결) 레이어로 구성됩니다.
# 각 레이어에는 활성화 함수로는 ReLU(Rectified Linear Unit)가 사용되고,
# 마지막 레이어에는 출력 클래스에 해당하는 3개의 뉴런과 softmax 활성화 함수가 사용됩니다.

# 모델은 categorical_crossentropy 손실 함수를 사용하여 컴파일되며, 옵티마이저로는 Adam이 사용됩니다.
# 훈련은 500번의 에포크를 수행하며, 배치 크기는 32로 설정됩니다.

# 훈련이 완료되면 모델은 테스트 세트에서 평가되고,
# 손실(loss), 평균 제곱 오차(mse), 정확도(accuracy)가 출력됩니다.
# 여기서 정확도는 1.0으로 출력되는데,
# 이는 모델이 테스트 세트에 있는 모든 샘플을 100% 정확하게 분류했음을 의미합니다.