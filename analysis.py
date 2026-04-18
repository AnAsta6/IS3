import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve

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

    def learning_curve(model, X, y, title="Learning Curve"):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, scoring='accuracy', random_state=42
        )

        # Средние значения
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)

        # Рисуем
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_mean, 'o-', label='Train', color='blue')
        plt.plot(train_sizes, val_mean, 'o-', label='Validation', color='green')

        plt.title(title)
        plt.xlabel('Количество обучающих примеров')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Простой вывод
        print(f"\nАнализ {title}:")
        print(f"  Train accuracy: {train_mean[-1]:.4f}")
        print(f"  Valid accuracy: {val_mean[-1]:.4f}")
        print(f"  Разница: {train_mean[-1] - val_mean[-1]:.4f}")

        gap = train_mean[-1] - val_mean[-1]

        if gap > 0.10:
            print("\nЕСТЬ ПЕРЕОБУЧЕНИЕ")
            print("   Точность на обучении значительно выше, чем на валидации.")
            print("   Рекомендации:")
            print("   - Упростить модель (уменьшить max_depth, увеличить min_samples_split)")
            print("   - Собрать больше обучающих данных")
            print("   - Добавить регуляризацию")

        elif gap > 0.05:
            print("\nНЕБОЛЬШОЕ ПЕРЕОБУЧЕНИЕ")
            print("   Модель слегка переобучена, но в целом работает хорошо.")
            print("   Рекомендации:")
            print("   - Можно попробовать упростить модель")
            print("   - Или собрать ещё немного данных")

        else:
            print("\nПЕРЕОБУЧЕНИЯ НЕТ")
            print("   Разница между train и validation минимальна.")
            print("   Модель хорошо обобщает новые данные.")

            if val_mean[-1] > 0.90:
                print("\nМОДЕЛЬ ДООБУЧЕНА")
                print("   Validation accuracy > 90% — отличный результат.")
                print("   Дополнительные данные не требуются.")
                print("\n   Рекомендации по улучшению:")
                print("   - Модель уже работает почти идеально")
                print("   - Можно использовать в продакшене без изменений")
                print("   - Дальнейшее улучшение маловероятно")
            else:
                print("\nМОДЕЛЬ НЕ ДООБУЧЕНА")
                print("   Validation accuracy < 90% — есть куда расти.")
                print("\n   Рекомендации по улучшению:")
                print("   - Собрать больше обучающих данных")
                print("   - Попробовать более сложную модель")
                print("   - Добавить новые признаки")

