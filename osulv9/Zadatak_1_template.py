import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

# učitaj CIFAR-10 skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikaži nekoliko slika
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]), plt.yticks([])
    plt.imshow(X_train[i])
plt.show()

# skaliranje
X_train_n = X_train.astype('float32') / 255.0
X_test_n = X_test.astype('float32') / 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# definiraj CNN model s Dropout slojevima
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # novi dropout sloj
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # novi dropout sloj

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # novi dropout sloj

    layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dropout(0.5),  # dropout prije izlaza

    layers.Dense(10, activation='softmax')
])

model.summary()

# TensorBoard callback u novi direktorij
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir='logs/cnn_dropout', update_freq=100),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# treniranje
model.fit(X_train_n,
          y_train,
          epochs=20,
          batch_size=64,
          callbacks=my_callbacks,
          validation_split=0.1)

# evaluacija
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Točnost na testnom skupu: {100.0 * score[1]:.2f}%')
