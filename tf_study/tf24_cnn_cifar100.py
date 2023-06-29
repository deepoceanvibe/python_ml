from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

# 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# # 시각화
# plt.imshow(x_train[0])
# plt.show()

# scaling  (이미지 0~255 -> 0~1 범위로 만들어줌)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

# 모델구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Conv2D(64, (2, 2)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='softmax'))

# 컴파일훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train,
          validation_split=0.2,
          callbacks=[earlyStopping],
          epochs=20, batch_size=128
          )
end_time = time.time() - start_time

# 평가예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('걸린시간 : ', end_time)