import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(page_title="Data Analysis & ML App", layout="wide")

# Title
st.title("📊 Interactive Data Analysis and Simple Machine Learning")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Read Data
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset Summary
    st.subheader("📌 Data Summary")
    st.write(df.describe(include='all'))

    # Data Info
    st.subheader("🧾 Data Information")
    buffer = df.info(buf=None)
    st.text(buffer)

    # Missing Values
    st.subheader("❗ Missing Values")
    st.dataframe(df.isnull().sum())

    # Data Types and Variables
    st.subheader("🧩 Variable Types")
    st.write(df.dtypes)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = df[col].astype(str)  # Ensure all are strings
        df[col] = le.fit_transform(df[col])

    st.sidebar.header("🔎 Visualization Controls")

    # Univariate
    st.sidebar.subheader("Univariate Analysis")
    univariate_col = st.sidebar.selectbox("Select column", df.columns)
    if st.sidebar.button("Plot Univariate"):
        st.subheader(f"Histogram / Count Plot of {univariate_col}")
        fig, ax = plt.subplots()
        if df[univariate_col].dtype == "object" or len(df[univariate_col].unique()) < 10:
            sns.countplot(x=df[univariate_col], ax=ax)
        else:
            sns.histplot(df[univariate_col], kde=True, ax=ax)
        st.pyplot(fig)

    # Bivariate
    st.sidebar.subheader("Bivariate Analysis")
    bivariate_x = st.sidebar.selectbox("X-axis", df.columns, key="biv_x")
    bivariate_y = st.sidebar.selectbox("Y-axis", df.columns, key="biv_y")
    if st.sidebar.button("Plot Bivariate"):
        st.subheader(f"Scatter/Box Plot of {bivariate_x} vs {bivariate_y}")
        fig, ax = plt.subplots()
        if df[bivariate_x].dtype != 'object' and df[bivariate_y].dtype != 'object':
            sns.scatterplot(x=df[bivariate_x], y=df[bivariate_y], ax=ax)
        else:
            sns.boxplot(x=df[bivariate_x], y=df[bivariate_y], ax=ax)
        st.pyplot(fig)

    # Multivariate
    st.sidebar.subheader("Multivariate Analysis")
    if st.sidebar.button("Correlation Heatmap"):
        st.subheader("🔗 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Regression
    st.subheader("📈 Simple Linear Regression")
    regression_target = st.selectbox("Select target variable", df.columns)
    regression_features = st.multiselect("Select independent variable(s)", df.columns.drop(regression_target))
    if st.button("Run Regression"):
        X = df[regression_features]
        y = df[regression_target]

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        st.write("R² Score:", r2_score(y, y_pred))
        st.write("Coefficients:", dict(zip(regression_features, model.coef_)))
        st.write("Intercept:", model.intercept_)

    # ML Suggestion
    st.subheader("🤖 Machine Learning Model Suggestion")
    target_col = st.selectbox("Select target for ML model", df.columns, key="ml_target")
    if st.button("Suggest ML Model"):
        if df[target_col].nunique() <= 10:
            st.write("🟢 Classification Problem Detected")
            model = RandomForestClassifier()
        else:
            st.write("🔵 Regression Problem Detected")
            model = RandomForestRegressor()

        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if isinstance(model, RandomForestClassifier):
            acc = accuracy_score(y_test, y_pred.round())
            st.write("✅ Accuracy Score:", acc)
        else:
            r2 = r2_score(y_test, y_pred)
            st.write("📈 R² Score:", r2)
