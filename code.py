import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from imblearn.over_sampling import SMOTE

class code:
    def load_data():

        wine = load_wine() #датасет Wine
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        return df, wine.target_names.tolist()

    def smote(X_train, y_train, k=5):
        sm = SMOTE(random_state=42, k_neighbors=k)
        return sm.fit_resample(X_train, y_train)