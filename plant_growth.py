import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('plant_growth_data.csv')

df.head()


df.tail()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
X = df.drop(['Growth_Milestone'], axis=1)
y = df['Growth_Milestone']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Before Oversampling:", y_train.value_counts())

ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
print("After Oversampling:", y_train_res.value_counts())

X_train_res = pd.get_dummies(X_train_res, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
# Ensure both train and test have the same columns
X_test = X_test.reindex(columns=X_train_res.columns, fill_value=0)

log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
log_reg.fit(X_train_res, y_train_res)

y_pred = log_reg.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn import svm

svc = svm.SVC(kernel = 'linear')

X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)
# Ensure both train and test have the same columns
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
svc.fit(X_train_encoded, y_train)

y_pred_svc = svc.predict(X_test_encoded)

print("\nSVC Accuracy:", accuracy_score(y_test, y_pred_svc))
print("\nSVC Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))
print("\nSVC Classification Report:\n", classification_report(y_test, y_pred_svc))

from sklearn.tree import DecisionTreeClassifier
# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
# Apply one-hot encoding to X_train and X_test
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)
# Ensure both train and test have the same columns
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
# Train the classifier
clf.fit(X_train_encoded, y_train)
# Make predictions
y_pred = clf.predict(X_test_encoded)
# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))