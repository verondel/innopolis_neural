
# Проект классификации изображений CIFAR-10 и CIFAR-100
## Описание
Этот проект посвящен задаче классификации цветных изображений из датасетов CIFAR-10 и CIFAR-100 с использованием свёрточных нейронных сетей (CNN). Основная цель - обучить модель, способную эффективно классифицировать изображения на основе их визуального содержания.

## Задача
1. Обучить CNN для классификации изображений CIFAR-10.
2. Модифицировать архитектуру для работы с CIFAR-100, классифицируя изображения на 100 узких и 20 широких классов.
3. Исследовать влияние различий между узкими и широкими классами на точность классификации.

## Структура репозитория
*  **model_training.ipynb**: Jupyter Notebook с кодом для обучения моделей и визуализации результатов. 
* **model_training.py**: скрипт Python для обучения моделей 
* **models/**: Папка, содержащая обученные модели (model_fine.keras и model_coarse.keras). 


## Использование
* Открыть main.ipynb для просмотра процесса обучения и анализа.
* Запустить main_2.py для повторения обучения моделей.


## Требования
* TensorFlow
* Matplotlib
* NumPy
* scikit-learn
