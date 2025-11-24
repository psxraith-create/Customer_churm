"""
Model training module.

Supports:
- Train-test split
- Training a classifier
- Saving model artifacts
"""

import pickle
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def split_data(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset into train and test sets."""
        X = df.drop(columns=[label_col])
            y = df[label_col]
               return train_test_split(X, y, test_size=0.2, random_state=42)


                def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
                    """Trains a RandomForest classifier."""
                        model = RandomForestClassifier(n_estimators=150, random_state=42)
                            model.fit(X_train, y_train)
                                return model


                                def save_model(model, filepath: str) -> None:
                                    """Saves trained model as a pickle file."""
                                        with open(filepath, "wb") as f:
                                                pickle.dump(model, f)