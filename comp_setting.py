from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

class setting():

    def create_pipeline():
        return Pipeline([
            ('scaler', StandardScaler()),  # Шаг масштабирования
            ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))  # Модель
        ])

    def grid_search(pipeline, X, y):
        #Сетка гиперпараметров
        param_grid = {
            'clf__max_depth': [5, 10, None],  # None означает "без ограничения глубины"
            'clf__n_estimators': [50, 100]
        }
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring='f1_macro',
            verbose=2,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        return grid_search


