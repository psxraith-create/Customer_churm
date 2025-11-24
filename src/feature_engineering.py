"""
Feature engineering utilities.

Creates:
- New interaction features
- Domain-specific engineered fields
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates new feature columns."""
        df = df.copy()

            # Example engineered features
                if {"TotalCharges", "MonthlyCharges"} <= set(df.columns):
                        df["AvgChargeRatio"] = df["TotalCharges"] / (df["MonthlyCharges"] + 1)

                            return df