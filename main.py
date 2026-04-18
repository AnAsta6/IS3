from code import code
from models import Models
from analysis import analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pandas as pd

if __name__ == '__main__':
    print("♡‧₊˚✧ Интеллектуальные системы. Практическое занятие № 3 ✧˚₊‧♡\n")

    #датасет
    df, target_names = code.load_data()
    print(f"1. Количество классов: {len(target_names)}")
    print(f"   Названия классов: {target_names}")
    print(f"   Количество признаков: {df.shape[1] - 1}\n")

    #EDA
    print("2. Распределение классов:")
    print(df['target'].value_counts().sort_index())
    print("   → Небольшой дисбаланс (класс 2 чуть меньше)\n")

    #разделка данных
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    #в модель пихаем и она типа умной становитьсч
    models = Models(X_train, X_test, y_train, y_test)

    print("♡‧₊˚✧Обучение и оценка моделей✧˚₊‧♡\n")

    #обычный рандомфарест
    acc_rf = models.rf_model()
    print("\nПолный отчёт:")
    print(classification_report(y_test, models.y_rf_pred, target_names=target_names, digits=4))
    analysis.confusion_matrix(y_test, models.y_rf_pred, "RandomForest", target_names)

    #LogisticRegression
    acc_lr = models.logisticRegression_model()
    print("\nПолный отчёт:")
    print(classification_report(y_test, models.y_lr_pred, target_names=target_names, digits=4))
    analysis.confusion_matrix(y_test, models.y_lr_pred, "LogisticRegression", target_names)

    #RandomForest и class_weight='balanced'
    acc_bal = models.rf_balanced_model()
    print("\nПолный отчёт: ")
    print(classification_report(y_test, models.y_rf_bal_pred, target_names=target_names, digits=4))
    analysis.confusion_matrix(y_test,  models.y_rf_bal_pred, "RandomForest (balanced)", target_names)

    #сравнение макро F1
    macro_f1_rf = f1_score(y_test, models.y_rf_pred, average='macro')
    macro_f1_lr = f1_score(y_test, models.y_lr_pred, average='macro')
    macro_f1_bal = f1_score(y_test, models.y_rf_bal_pred, average='macro')

    print(f"\nMacro avg F1-score:")
    print(f"   RandomForest:          {macro_f1_rf:.4f}")
    print(f"   LogisticRegression:    {macro_f1_lr:.4f}")
    print(f"   RandomForest (balanced): {macro_f1_bal:.4f}")

    #анализ худшего класса (по обычному RandomForest)
    report_dict = classification_report(y_test, models.y_rf_pred, target_names=target_names, output_dict=True)
    recalls = {cls: report_dict[cls]['recall'] for cls in target_names}
    worst_class = min(recalls, key=recalls.get)
    print(f"\nХудший класс у обычного RandomForest — {worst_class} (recall = {recalls[worst_class]:.4f})")

    print("\n♡‧₊˚✧ВЫВОД✧˚₊‧♡")
    print("Лучше всего с балансом классов справляется RandomForest с class_weight='balanced'.")
    print("Recall для самого слабого класса вырос, macro F1 тоже улучшился.")
    print("Для итогового прототипа выбираю: RandomForestClassifier(class_weight='balanced')")

    """
    То что тут много едениц и везде еденицы-это норм.
    Это происходит из-за датасета Wine. эта штука дает cлишком чистые данные
    """

    #График точности
    analysis.graf(['RandomForest', 'LogisticRegression', 'RF balanced'],
                  [acc_rf, acc_lr, acc_bal])

    #________Шаг 2________
    print("SMOTE трансформация")
    X_resampled, y_resampled=code.smote(X_train, y_train)
    new_models = Models(X_resampled, X_test, y_resampled, y_test)
    acc_rf = new_models.rf_model()
    print("\nПолный отчёт:")
    print(classification_report(y_test, new_models.y_rf_pred, target_names=target_names, digits=4))
    analysis.confusion_matrix(y_test, new_models.y_rf_pred, "new_RandomForest", target_names)
    new_macro_f1_rf = f1_score(y_test, new_models.y_rf_pred, average='macro')
    print(f"\nMacro avg F1-score:")
    print(f"   New_RandomForest:          {new_macro_f1_rf:.4f}")
    print(f"   Разница RandomForest:          {macro_f1_rf-new_macro_f1_rf:.4f}")
    analysis.learning_curve(new_models.rf_model_obj, X_train, y_train, title="Learning Curve")

    print("\n♡‧₊˚✧Готово!✧˚₊‧♡")