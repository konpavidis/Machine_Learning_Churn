import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Total number of samples
total_samples = len(data)

# Start by preprocessing the data
# Convert totalcharges to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
# For missing values
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

# Encode categorical variables from the columns provided
label_encoder = LabelEncoder()
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod', 'Churn']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Define features and target variable
X = data.drop(columns=['customerID', 'Churn'])
y = data['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Load Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

with open("model_evaluation_results.txt", "w") as file:
    file.write("Total number of samples: {}\n".format(total_samples))
    file.write("\nAccuracy: {}\n".format(accuracy))
    file.write("Confusion Matrix:\n{}\n".format(conf_matrix))
    file.write("Classification Report:\n{}\n".format(report))

