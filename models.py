from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class Models:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_rf_pred = None
        self.y_rf_bal_pred = None
        self.y_lr_pred = None
        self.rf_model_obj = None
        self.rf_bal_model_obj = None
        self.lr_model_obj = None

    def rf_model(self):
        print("RandomForestClassifier модель (по умолчанию)")
        self.rf_model_obj = RandomForestClassifier(random_state=42)
        self.rf_model_obj.fit(self.X_train, self.y_train)

        y_train_pred = self.rf_model_obj.predict(self.X_train)
        self.y_rf_pred = self.rf_model_obj.predict(self.X_test)

        accuracy_train = accuracy_score(self.y_train, y_train_pred)
        accuracy_test = accuracy_score(self.y_test, self.y_rf_pred)

        print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_train:.4f}")
        print("\nТЕСТОВАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_test:.4f}")
        return accuracy_test
    def rf_balanced_model(self):
        print("RandomForestClassifier модель (class_weight='balanced')")
        self.rf_bal_model_obj = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.rf_bal_model_obj.fit(self.X_train, self.y_train)

        y_train_pred = self.rf_bal_model_obj.predict(self.X_train)
        self.y_rf_bal_pred = self.rf_bal_model_obj.predict(self.X_test)

        accuracy_train = accuracy_score(self.y_train, y_train_pred)
        accuracy_test = accuracy_score(self.y_test, self.y_rf_bal_pred)

        print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_train:.4f}")
        print("\nТЕСТОВАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_test:.4f}")
        return accuracy_test

    def logisticRegression_model(self):
        print("LogisticRegression модель")
        self.lr_model_obj= StandardScaler()
        X_train_scaled =self.lr_model_obj.fit_transform(self.X_train)
        X_test_scaled =self.lr_model_obj.transform(self.X_test)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, self.y_train)

        y_train_pred =model.predict(X_train_scaled)
        self.y_lr_pred =model.predict(X_test_scaled)

        accuracy_train = accuracy_score(self.y_train, y_train_pred)
        accuracy_test = accuracy_score(self.y_test, self.y_lr_pred)

        print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_train:.4f}")
        print("\nТЕСТОВАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_test:.4f}")
        return accuracy_test