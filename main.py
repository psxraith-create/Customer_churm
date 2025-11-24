"""
Main pipeline for Customer Churn Prediction project.

This script:
1. Loads raw dataset
2. Cleans data
3. Performs feature engineering
4. Splits train/test
5. Trains ML model
6. Evaluates model
7. Saves model + artifacts

Written with production-level structure for portfolio demonstration.
"""

import logging
import os
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import engineer_features
from src.train_model import train_model, save_model
from src.evaluate_model import evaluate_model
from src.utils import ensure_directories


# -----------------------------
# CONFIGURE LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
        )

        RAW_DATA_PATH = "data/raw/customer_churn.csv"
        PROCESSED_DATA_PATH = "data/processed/cleaned_churn.csv"
        MODEL_OUTPUT_PATH = "models/churn_model.pkl"


        def main():
            """Execute the full churn prediction pipeline."""

                logging.info("📌 Ensuring required directories exist...")
                    ensure_directories(["data/raw", "data/processed", "models", "notebooks"])

                        # ----------------------------------------------------
                            # 1. LOAD RAW DATA
                                # ----------------------------------------------------
                                    if not os.path.exists(RAW_DATA_PATH):
                                            logging.error(
                                                        f"❌ Raw dataset not found at {RAW_DATA_PATH}. "
                                                                    "Please upload customer_churn.csv into data/raw/"
                                                                            )
                                                                                    return

                                                                                        logging.info("📥 Loading raw dataset...")
                                                                                            df = load_data(RAW_DATA_PATH)

                                                                                                # ----------------------------------------------------
                                                                                                    # 2. CLEANING DATA
                                                                                                        # ----------------------------------------------------
                                                                                                            logging.info("🧹 Cleaning dataset...")
                                                                                                                df_cleaned = clean_data(df)

                                                                                                                    logging.info(f"💾 Saving cleaned data → {PROCESSED_DATA_PATH}")
                                                                                                                        df_cleaned.to_csv(PROCESSED_DATA_PATH, index=False)

                                                                                                                            # ----------------------------------------------------
                                                                                                                                # 3. FEATURE ENGINEERING
                                                                                                                                    # ----------------------------------------------------
                                                                                                                                        logging.info("⚙️ Engineering features...")
                                                                                                                                            df_features = engineer_features(df_cleaned)

                                                                                                                                                # ----------------------------------------------------
                                                                                                                                                    # 4. TRAIN/TEST SPLIT
                                                                                                                                                        # ----------------------------------------------------
                                                                                                                                                            X_train, X_test, y_train, y_test = split_data(df_features)

                                                                                                                                                                # ----------------------------------------------------
                                                                                                                                                                    # 5. TRAIN MODEL
                                                                                                                                                                        # ----------------------------------------------------
                                                                                                                                                                            logging.info("🤖 Training machine learning model...")
                                                                                                                                                                                model = train_model(X_train, y_train)

                                                                                                                                                                                    logging.info(f"💾 Saving trained model → {MODEL_OUTPUT_PATH}")
                                                                                                                                                                                        save_model(model, MODEL_OUTPUT_PATH)

                                                                                                                                                                                            # ----------------------------------------------------
                                                                                                                                                                                                # 6. EVALUATE MODEL
                                                                                                                                                                                                    # ----------------------------------------------------
                                                                                                                                                                                                        logging.info("📊 Evaluating model...")
                                                                                                                                                                                                            evaluate_model(model, X_test, y_test)

                                                                                                                                                                                                                logging.info("✅ Pipeline completed successfully!")


                                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                                    main()