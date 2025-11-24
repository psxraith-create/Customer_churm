"""
Model evaluation module.

Generates:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
"""

from typing import Dict
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluates classifier performance."""
        predictions = model.predict(X_test)

            metrics = {
                    "accuracy": accuracy_score(y_test, predictions),
                            "classification_report": classification_report(
                                        y_test, predictions, output_dict=True
                                                ),
                                                    }

                                                        return metrics