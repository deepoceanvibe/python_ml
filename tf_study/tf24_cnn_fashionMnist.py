from keras.datasets import fashion_mnist
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping
import time

# 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

# # 시각화
# plt.imshow(x_train[98], 'gray')
# plt.show()

# [실습] Conv2D 2개이상 사용, Dropout 사용, Maxpooling2D 사용, accuracy = 0.9 이상

# 모델
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4),
                 padding='same',
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2, 2),
                 padding='valid',
                 activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# 컴파일훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    restore_best_weights=True
)
start_time = time.time()
model.fit(x_train, y_train,
          validation_split=0.2,
          callbacks=[earlyStopping],
          epochs=50,
          batch_size=32,
          verbose=1)
end_time = time.time() - start_time

# 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('걸린시간 : ', end_time)