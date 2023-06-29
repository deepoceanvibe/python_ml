from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Conv2D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

# 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# plt.imshow(x_train[5])
# plt.show()

# scaling
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

# 모델구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.7))
model.add(Conv2D(32, (3, 3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(42, activation='softmax'))

# 컴파일훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(
    monitor='val loss',
    mode='min',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(
    x_train, y_train,
    validation_split=0.2,
    callbacks=[earlyStopping],
    epochs=50,
    batch_size=32
)
end_time = time.time() - start_time

# 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('걸린시간 : ', end_time)