import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
import warnings

warnings.filterwarnings('ignore')

# Title of the app
st.title("🌱 Plant Growth Prediction")

# Load the dataset
st.subheader("Dataset Preview")
df = pd.read_csv('plant_growth_data.csv')
st.write(df.head())

# Features and Target
X = df.drop(['Growth_Milestone'], axis=1)
y = df['Growth_Milestone']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.write("Before Oversampling:")
st.write(y_train.value_counts())

# Apply Random Over Sampler
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

st.write("After Oversampling:")
st.write(y_train_res.value_counts())

# One-hot encoding
X_train_res = pd.get_dummies(X_train_res, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X_train_res.columns, fill_value=0)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
log_reg.fit(X_train_res, y_train_res)
y_pred_log = log_reg.predict(X_test)

st.subheader("🔍 Logistic Regression Results")
st.write("Accuracy:", accuracy_score(y_test, y_pred_log))
st.text("Classification Report:\n" + classification_report(y_test, y_pred_log))

# SVM
svc = svm.SVC(kernel='linear')
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
svc.fit(X_train_encoded, y_train)
y_pred_svc = svc.predict(X_test_encoded)

st.subheader("🔍 SVM Results")
st.write("Accuracy:", accuracy_score(y_test, y_pred_svc))
st.text("Classification Report:\n" + classification_report(y_test, y_pred_svc))

# Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_encoded, y_train)
y_pred_dt = clf.predict(X_test_encoded)

st.subheader("🔍 Decision Tree Results")
st.write("Accuracy:", accuracy_score(y_test, y_pred_dt))
st.text("Classification Report:\n" + classification_report(y_test, y_pred_dt))
