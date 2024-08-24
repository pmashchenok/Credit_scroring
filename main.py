import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score


def print_metrics(y_valid, predictions):
    accuracy = accuracy_score(y_valid, predictions)
    precision = precision_score(y_valid, predictions)
    recall = recall_score(y_valid, predictions)
    f1 = f1_score(y_valid, predictions)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


def show_cm(y_valid, predictions):
    target_names = ["Charged Off", "Fully Paid"]
    cm = confusion_matrix(y_valid, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues')
    plt.show()


df = pd.read_csv("credit_train.csv")

y = df["Loan Status"]
X = df.drop(columns=['Loan ID', 'Customer ID', 'Loan Status'])
X = X.iloc[:10000]
y = y.iloc[:10000]

# Data Cleaning
X.Purpose = X.Purpose.replace({"other": "Other", "moving": "Personal",
                               "Educational Expenses": "Personal", "wedding": "Personal",
                               "Medical Bills": "Personal", "Buy a Car": "Personal",
                               "Buy House": "Personal", "Home Improvements": "Personal",
                               "Take a Trip": "Personal", "vacation": "Personal",
                               "major_purchase": "Personal", "small_business": "Personal"})

X["Home Ownership"] = X["Home Ownership"].replace({"HaveMortgage": "Home Mortgage"})

X["Years in current job"] = X["Years in current job"].replace({"7 years": "7-9 years",
                                                               "8 years": "7-9 years",
                                                               "9 years": "7-9 years",
                                                               "4 years": "4-6 years",
                                                               "5 years": "4-6 years",
                                                               "6 years": "4-6 years",
                                                               "< 1 year": "0-1 year",
                                                               "1 year": "0-1 year",
                                                               "2 years": "2-3 years",
                                                               "3 years": "2-3 years"})


# Missing Data
X = X.drop(columns=['Months since last delinquent', 'Bankruptcies', 'Tax Liens'])
imputer = SimpleImputer(strategy='mean')
X[['Credit Score', 'Annual Income']] = imputer.fit_transform(X[['Credit Score', 'Annual Income']])

nan_cols = ['Years in current job']
imputer = SimpleImputer(strategy='most_frequent')
X[nan_cols] = imputer.fit_transform(X[nan_cols])

# Encoding Target Columns
pd.set_option('future.no_silent_downcasting', True)
y.replace({"Fully Paid": 1, "Charged Off": 0}, inplace=True)
y = y.astype(int)

# Preprocessing Numerical Columns
scaling_cols = [col for col in X.columns if X[col].dtype in ['float64', 'int32']]
scaling_cols.remove('Number of Credit Problems')
sc = StandardScaler()
X[scaling_cols] = sc.fit_transform(X[scaling_cols])

# Encoding Categorical Columns
obj_cols = [col for col in X.columns if X[col].dtype == "object"]
encoder = LabelEncoder()
for col in obj_cols:
    X[col] = encoder.fit_transform(X[col])


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2)

# Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print("Random Forest Classifier")
print(classification_report(y_valid, predictions))
show_cm(y_valid, predictions)
print_metrics(y_valid, predictions)

# Feature Importance
feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df.set_index('Feature', inplace=True)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df.index, feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Feature Importance')
plt.show()

# Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print("Logistic Regression")
print(classification_report(y_valid, predictions))
show_cm(y_valid, predictions)
print_metrics(y_valid, predictions)

# SVM
model = SVC(kernel='poly', random_state=42, degree=2)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print("SVM")
print(classification_report(y_valid, predictions))
show_cm(y_valid, predictions)
print_metrics(y_valid, predictions)

# XGBClassifier
model = XGBClassifier(scale_pos_weight=3.5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print("XGBClassifier")
print(classification_report(y_valid, predictions))
show_cm(y_valid, predictions)
print_metrics(y_valid, predictions)
