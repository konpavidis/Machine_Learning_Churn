# Python ML for Churn

## Overview
Customer churn prediction is a crucial task for businesses aiming to retain their customers. This project focuses on building a predictive model to forecast customer churn using logistic regression. By analyzing customer data such as demographics, services subscribed, and contract details, the model aims to identify customers who are likely to churn, enabling proactive retention strategies.

## Installation
1. Clone the repository.
2. Install the required dependencies using the following command:
  pip install pandas scikit-learn

## Usage
1. Make sure you have Python installed.
2. Run the following script:
  python your_script_name.py

## Project Structure
- **main.py**: This script contains the main code for data preprocessing, model training, and evaluation.
- **WA_Fn-UseC_-Telco-Customer-Churn.csv**: The dataset used in the project.
- **Link to Kaggle to the dataset used: https://www.kaggle.com/datasets/blastchar/telco-customer-churn, Thanks BlastChar!**

## Data Preprocessing
- The dataset is loaded using pandas.
- Missing values in the 'TotalCharges' column are filled with the mean value.
- Categorical variables are encoded using LabelEncoder.

## Model Training and Evaluation
- Features and target variable are defined.
- The data is split into training and testing sets.
- Features are standardized using StandardScaler.
- Logistic regression model is trained on the training data.
- Model predictions are made on the testing data.
- Model accuracy, confusion matrix, and classification report are evaluated and saved in "model_evaluation_results.txt".

## Results
- "model_evaluation_results.txt" contains the evaluation metrics of the trained model.

## Contributors
- Konstantinos Pavlakis (https://github.com/konpavidis)

