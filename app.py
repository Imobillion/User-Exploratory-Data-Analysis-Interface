import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import io

# App title and description
st.set_page_config(page_title="Data Exploration App", layout="wide")
st.title("📊 User-Interactive Data Visualization & Regression App")
st.markdown("Upload your dataset and perform **univariate, bivariate, multivariate analysis** and **simple linear regression** interactively.")

# Upload data
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    # Load the data
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("🧾 Data Summary")
    st.write(df.describe(include='all'))

    st.subheader("📋 Data Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("❓ Missing Values")
    st.write(df.isnull().sum())

    st.subheader("🔎 Variable Types")
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical = df.select_dtypes(include=['number']).columns.tolist()
    st.write(f"**Categorical Variables:** {categorical}")
    st.write(f"**Numerical Variables:** {numerical}")

    # Analysis sections
    st.header("📌 1. Univariate Analysis")
    uni_column = st.selectbox("Select a column", df.columns)

    if uni_column in numerical:
        st.write("Histogram")
        fig, ax = plt.subplots()
        sns.histplot(df[uni_column].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

        st.write("Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[uni_column], ax=ax)
        st.pyplot(fig)

    elif uni_column in categorical:
        st.write("Bar Chart")
        fig, ax = plt.subplots()
        df[uni_column].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    st.header("📌 2. Bivariate Analysis")
    col1 = st.selectbox("X Variable", df.columns, key='biv1')
    col2 = st.selectbox("Y Variable", df.columns, key='biv2')

    if col1 in numerical and col2 in numerical:
        st.write("Scatter Plot with Regression Line")
        fig, ax = plt.subplots()
        sns.regplot(x=df[col1], y=df[col2], ax=ax)
        st.pyplot(fig)

        corr = df[[col1, col2]].corr().iloc[0, 1]
        st.write(f"Correlation: **{corr:.2f}**")

    elif col1 in categorical and col2 in numerical:
        st.write("Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col1], y=df[col2], ax=ax)
        st.pyplot(fig)

    elif col1 in numerical and col2 in categorical:
        st.write("Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col2], y=df[col1], ax=ax)
        st.pyplot(fig)

    st.header("📌 3. Multivariate Analysis")
    multivars = st.multiselect("Select multiple numerical variables", numerical)

    if len(multivars) >= 2:
        st.write("Pairplot")
        fig = sns.pairplot(df[multivars].dropna())
        st.pyplot(fig)

        st.write("Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[multivars].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.header("📌 4. Simple Linear Regression")
    reg_x = st.selectbox("Select Independent Variable (X)", numerical, key="reg_x")
    reg_y = st.selectbox("Select Dependent Variable (Y)", numerical, key="reg_y")

    if reg_x and reg_y:
        X = df[[reg_x]].dropna()
        y = df[reg_y].dropna()
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        r2 = model.score(X, y)

        st.write(f"**Regression Equation:** {reg_y} = {model.intercept_:.2f} + {model.coef_[0]:.2f} * {reg_x}")
        st.write(f"**R-squared:** {r2:.3f}")

        fig, ax = plt.subplots()
        ax.scatter(X, y, label="Actual")
        ax.plot(X, y_pred, color="red", label="Predicted")
        ax.set_xlabel(reg_x)
        ax.set_ylabel(reg_y)
        ax.legend()
        st.pyplot(fig)

        # Optional: Show statsmodels summary
        if st.checkbox("Show detailed regression summary (Statsmodels)"):
            X_const = sm.add_constant(X)
            model_sm = sm.OLS(y, X_const).fit()
            st.text(model_sm.summary())
else:
    st.info("👈 Please upload a CSV file to begin analysis.")
