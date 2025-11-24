"""
data_preprocessing.py

Handles all data cleaning for the Customer Churn Prediction project.
Includes:
- Missing value handling
- Categorical encoding
- Numeric scaling
- Train/Test preparation

Author: Priyangshu Sarkar 
"""

from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
        Clean raw customer churn dataset.

            Steps:
                    - Strip column names
                            - Convert TotalCharges to numeric
                                    - Handle missing values
                                            - Drop customerID (not useful)

                                                Args:
                                                        df: Raw dataframe loaded from CSV.

                                                            Returns:
                                                                    Cleaned dataframe ready for preprocessing.
                                                                        """

                                                                            logger.info("Starting raw data cleaning...")

                                                                                # Strip spaces
                                                                                    df.columns = df.columns.str.strip()

                                                                                        # Convert data type
                                                                                            if "TotalCharges" in df.columns:
                                                                                                    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

                                                                                                        # Drop ID column
                                                                                                            if "customerID" in df.columns:
                                                                                                                    df = df.drop("customerID", axis=1)

                                                                                                                        # Handle missing
                                                                                                                            df = df.dropna()

                                                                                                                                logger.info("Raw data cleaned successfully.")
                                                                                                                                    return df


                                                                                                                                    def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
                                                                                                                                        """
                                                                                                                                            Create preprocessing pipeline for numeric + categorical features.

                                                                                                                                                Args:
                                                                                                                                                        df: Cleaned dataframe.

                                                                                                                                                            Returns:
                                                                                                                                                                    ColumnTransformer object for ML pipeline.
                                                                                                                                                                        """

                                                                                                                                                                            logger.info("Building preprocessing pipeline...")

                                                                                                                                                                                numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                                                                                                                                                                                    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

                                                                                                                                                                                        # Remove target column if present
                                                                                                                                                                                            if "Churn" in categorical_cols:
                                                                                                                                                                                                    categorical_cols.remove("Churn")

                                                                                                                                                                                                        numeric_pipeline = Pipeline(steps=[
                                                                                                                                                                                                                ("scaler", StandardScaler())
                                                                                                                                                                                                                    ])

                                                                                                                                                                                                                        categorical_pipeline = Pipeline(steps=[
                                                                                                                                                                                                                                ("encoder", OneHotEncoder(handle_unknown="ignore"))
                                                                                                                                                                                                                                    ])

                                                                                                                                                                                                                                        preprocessor = ColumnTransformer(
                                                                                                                                                                                                                                                transformers=[
                                                                                                                                                                                                                                                            ("num", numeric_pipeline, numeric_cols),
                                                                                                                                                                                                                                                                        ("cat", categorical_pipeline, categorical_cols)
                                                                                                                                                                                                                                                                                ]
                                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                        logger.info("Preprocessor built successfully.")
                                                                                                                                                                                                                                                                                            return preprocessor


                                                                                                                                                                                                                                                                                            def create_train_test(
                                                                                                                                                                                                                                                                                                df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2, random_state: int = 42
                                                                                                                                                                                                                                                                                                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                                                                                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                                                                                        Split dataset into train and test sets.

                                                                                                                                                                                                                                                                                                            Args:
                                                                                                                                                                                                                                                                                                                    df: Cleaned dataframe.
                                                                                                                                                                                                                                                                                                                            target_col: Column name for the label.
                                                                                                                                                                                                                                                                                                                                    test_size: Test set ratio.
                                                                                                                                                                                                                                                                                                                                            random_state: Seed for reproducibility.

                                                                                                                                                                                                                                                                                                                                                Returns:
                                                                                                                                                                                                                                                                                                                                                        X_train, X_test, y_train, y_test
                                                                                                                                                                                                                                                                                                                                                            """

                                                                                                                                                                                                                                                                                                                                                                logger.info("Splitting dataset into train and test sets...")

                                                                                                                                                                                                                                                                                                                                                                    X = df.drop(target_col, axis=1)
                                                                                                                                                                                                                                                                                                                                                                        y = df[target_col].map({"Yes": 1, "No": 0})  # map Churn to 1/0

                                                                                                                                                                                                                                                                                                                                                                            X_train, X_test, y_train, y_test = train_test_split(
                                                                                                                                                                                                                                                                                                                                                                                    X, y,
                                                                                                                                                                                                                                                                                                                                                                                            test_size=test_size,
                                                                                                                                                                                                                                                                                                                                                                                                    random_state=random_state,
                                                                                                                                                                                                                                                                                                                                                                                                            stratify=y
                                                                                                                                                                                                                                                                                                                                                                                                                )

                                                                                                                                                                                                                                                                                                                                                                                                                    logger.info(
                                                                                                                                                                                                                                                                                                                                                                                                                            f"Dataset split completed. "
                                                                                                                                                                                                                                                                                                                                                                                                                                    f"Train samples: {len(X_train)}, Test samples: {len(X_test)}"
                                                                                                                                                                                                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                                                                                                                                                                                                            return X_train, X_test, y_train, y_test