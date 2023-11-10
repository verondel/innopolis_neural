from tensorflow.keras.models import load_model

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Загрузка и предобработка датасета CIFAR-10
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Создание архитектуры модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # 10 - количество классов в CIFAR-10
])

# Компиляция модели
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Обучение модели
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Создание архитектуры модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Компиляция модели
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Обучение модели
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Оценка производительности модели
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Точность:', test_acc)

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Загрузка и предобработка датасета CIFAR-100
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
train_images, test_images = train_images / 255.0, test_images / 255.0

# Преобразование меток в one-hot encoding для узких классов
train_labels_fine = to_categorical(train_labels, 100)
test_labels_fine = to_categorical(test_labels, 100)

# Загрузка меток широких классов
(train_images, train_labels_coarse), (test_images, test_labels_coarse) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')

# Преобразование меток в one-hot encoding для широких классов
train_labels_coarse = to_categorical(train_labels_coarse, 20)
test_labels_coarse = to_categorical(test_labels_coarse, 20)

# Создание архитектуры модели
def create_model(num_classes):
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(512, activation='relu'),
        Dense(100, activation='softmax'),
    ])
    return model

def create_model_2(num_classes):
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(512, activation='relu'),
        Dense(20, activation='softmax'),
    ])
    return model

# Создание модели для 100 узких классов
from tensorflow.keras import optimizers
model_fine = create_model(100)
model_fine.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])

# Обучение модели на узких классах
model_fine.fit(train_images, train_labels_fine, epochs=10, validation_data=(test_images, test_labels_fine))

# Создание модели для 20 широких классов
model_coarse = create_model_2(20)
model_coarse.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])

# Обучение модели на широких классах
model_coarse.fit(train_images, train_labels_coarse, epochs=10, validation_data=(test_images, test_labels_coarse))

# compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Сравнение производительности моделей
test_loss_fine, test_acc_fine = model_fine.evaluate(test_images, test_labels_fine, verbose=2)
test_loss_coarse, test_acc_coarse = model_coarse.evaluate(test_images, test_labels_coarse, verbose=2)


fine_accuracy = model_fine.evaluate(test_images, test_labels_fine, verbose=0)[1]
coarse_accuracy = model_coarse.evaluate(test_images, test_labels_coarse, verbose=0)[1]
print(f"Accuracy для узких классов: {test_acc_fine}")
print(f"Accuracy для широких классов: {coarse_accuracy}")

# Функция для вычисления точности для каждого класса
def calculate_class_accuracy(model, images, true_labels):
    # Получение предсказаний модели
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)

    # Получение отчета по классификации для каждого класса
    class_report = classification_report(true_classes, predicted_classes, output_dict=True)
    return class_report


import numpy as np
from sklearn.metrics import classification_report

# Вычисление точности для каждого узкого класса
class_accuracy_fine = calculate_class_accuracy(model_fine, test_images, test_labels_fine)

# Вычисление точности для каждого широкого класса
class_accuracy_coarse = calculate_class_accuracy(model_coarse, test_images, test_labels_coarse)

accuracy_diff = {} # Словарь для разности точностей
for fine_class in range(100):
    coarse_class = fine_class // 5 # Примерное сопоставление
    diff = class_accuracy_fine[str(fine_class)]['precision'] - class_accuracy_coarse[str(coarse_class)]['precision']
    accuracy_diff[fine_class] = diff

# Визуализация
plt.bar(accuracy_diff.keys(), accuracy_diff.values())
plt.xlabel('Классы')
plt.ylabel('Разность в точности')
plt.title('Разность точности между узкими и широкими классами')
plt.show()

model_fine.save('model_fine.keras')

model_coarse.save('model_coarse.keras')

