import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score

# 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape)      # (569, 30)
print(y.shape)      # (569,)
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1234,
    shuffle=True
)

print(x_train.shape, x_test.shape)      # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)      # (398,) (171,)


# 모델 구성 
model = Sequential()
model.add(Dense(65, input_dim=30))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid'))       # 이진분류는 마지막 아웃풋 레이어에 무조건 sigmoid 함수사용

# 컴파일훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['mse', 'accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=128)

# 평가예측
loss, mse, accuracy = model.evaluate(x_test, y_test)        # metrics를 걸었기 때문에 mse, accuracy까지 넣음
print('loss : ', loss)      # loss :  0.31242209672927856
print('mse : ', mse)        # mse :  0.07013959437608719
print('accuracy : ', accuracy)      # accuracy :  0.9181286692619324

y_predict = model.predict(x_test)
# print(y_predict)

# y_predict = np.where(y_predict > 0.5, 1, 0)
y_predict = np.round(y_predict)
print(y_predict)

acc_score = accuracy_score(y_test, y_predict)
print('acc_score : ', acc_score)