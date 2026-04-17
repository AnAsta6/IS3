import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class analysis:

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name, target_names=None):
        cm = confusion_matrix(y_true, y_pred)

        if target_names is None:
            target_names = ['class_0', 'class_1', 'class_2']

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(cmap='viridis', ax=ax, values_format='d')

        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказанные значения')
        plt.show()

        return cm

    @staticmethod #диаграмма точности моделей
    def graf(models, accuracy, colors=None):
        if colors is None:
            #зелёный-синий-фиолетовый
            colors = ['#4CAF50', '#2196F3', '#9C27B0']

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracy, color=colors, edgecolor='black', linewidth=1.2)
        plt.title("Диаграмма точности моделей")
        plt.xlabel("Модели")
        plt.ylabel("Точность (Accuracy)")
        plt.ylim(0, 1)
        plt.show()

