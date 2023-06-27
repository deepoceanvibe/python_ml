from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from keras.utils import to_categorical
import time

# 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (178, 13) (178,)
print(datasets.feature_names)       # ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
print(datasets.DESCR)               # class_0 class_1 class_2

# 원핫인코딩
y = to_categorical(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    test_size=0.2,
    random_state=278,
    shuffle=True
)

print(x_train.shape, x_test.shape)      # (124, 13) (54, 13)
print(y_train.shape, y_test.shape)      # (124, 3) (54, 3)


# 모델구성
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(3, activation='softmax'))


# 컴파일훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])
start_time = time.time()
hist = model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=500,
          batch_size=64)
end_time = time.time() - start_time

print('걸린 시간 : ', end_time)


# 평가예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mse : ', mse)
print('accuracy : ', accuracy)
print('걸린 시간 : ', end_time)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')

plt.title('Loss & Val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
