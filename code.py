import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

class code:
    def load_data():

        wine = load_wine() #датасет Wine
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        return df, wine.target_names.tolist()