practices.


---

📂 Project Structure

Customer_Churn/
│
├── data/
│   ├── raw/
│   │   └── customer_churn.csv
│   ├── processed/
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── utils.py
│
├── models/
│   └── model.pkl
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore


---

📌 Project Overview

Customer churn is a major challenge in banking, telecom, insurance, and subscription-based businesses.
The aim of this project is to:

Analyze customer data

Clean and preprocess it

Train machine learning models

Identify which customers are likely to churn

Provide actionable insights


The project is fully modular and can be plugged into any real-world customer dataset.


---

🎯 Objectives

Understand customer behavior patterns

Build a prediction model with high accuracy

Evaluate model performance using professional ML metrics

Provide reusable modules for future datasets



---

⚙️ Features

✔ Clean Data Pipeline – missing values, encoding, normalization
✔ Exploratory Data Analysis (EDA)
✔ Train multiple ML models (Random Forest, XGBoost, Logistic Regression)
✔ Automatic model selection based on accuracy
✔ Pickle-based model saving/loading
✔ Easily replace your dataset with any CSV
✔ Compatible with GitHub Codespaces and VS Code


---

🧠 Core Columns Required

Your dataset must contain these important columns:

Column	Description

customer_id	Unique ID of customer
credit_score	Score of customer
age	Age
gender	Male/Female
tenure	Duration with bank
balance	Bank balance
num_of_products	Number of products used
has_cr_card	Credit card existence
is_active_member	Active user flag
estimated_salary	Estimated salary
churn	Target variable (0/1)



---

🚀 Installation

1. Clone the repository

git clone https://github.com/psxraith-create/Customer_churm
cd Customer_churm

2. Install dependencies

pip install -r requirements.txt


---

📊 Running the Project

1. Ensure dataset is placed correctly

Place your dataset here:

data/raw/customer_churn.csv

2. Run the main script

python main.py

This will:

Load and clean the data

Train the model

Save the trained model into /models/model.pkl

Print evaluation metrics



---

🔍 Model Evaluation

The model is evaluated using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix


The terminal will show everything after running main.py.


---

📈 Improving the Model

To improve performance, you can:

Add more features

Perform hyperparameter tuning

Try advanced algorithms (XGBoost, CatBoost)

Balance dataset using SMOTE

Add cross-validation



---

🔁 Replacing the Dataset (Future Use)

You can replace customer_churn.csv with any other customer dataset.

Just maintain:

Same column names

Or update names inside data_preprocessing.py


Then re-run:

python main.py


---

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-Learn

Matplotlib, Seaborn

GitHub Codespaces

VS Code



---

📬 Contact

For queries or improvements:
Author: Priyangshu Sarkar
GitHub: https://github.com/psxraith-create
