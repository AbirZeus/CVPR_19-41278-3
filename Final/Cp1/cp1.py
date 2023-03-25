import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
rmsprop_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

model.compile(optimizer=sgd_optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_sgd = model.fit(train_images, train_labels, epochs=10,validation_data=(test_images, test_labels))

model.compile(optimizer=adam_optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_adam = model.fit(train_images, train_labels, epochs=10,validation_data=(test_images, test_labels))

model.compile(optimizer=rmsprop_optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_rmsprop = model.fit(train_images, train_labels, epochs=10,validation_data=(test_images, test_labels))

plt.plot(history_sgd.history['accuracy'], label='SGD train', color='blue')
plt.plot(history_sgd.history['val_accuracy'], label='SGD test', color='blue', linestyle='dashed')
plt.plot(history_adam.history['accuracy'], label='Adam train', color='red')
plt.plot(history_adam.history['val_accuracy'], label='Adam test', color='red', linestyle='dashed')
plt.plot(history_rmsprop.history['accuracy'], label='RMSprop train', color='green')
plt.plot(history_rmsprop.history['val_accuracy'], label='RMSprop test', color='green', linestyle='dashed')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()