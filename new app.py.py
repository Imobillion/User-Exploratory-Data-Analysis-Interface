import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix

# Page setup
st.set_page_config(page_title="📊 Data Analyzer & Auto ML", layout="wide")

st.title("📊 Smart Data Explorer with Auto ML")
st.markdown("Upload your dataset and get automated insights, visualization, and predictive modeling with explanation.")

st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Load Data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # === Section 1: Data Overview ===
    with st.expander("📂 1. Data Overview", expanded=True):
        st.subheader("🔍 Preview")
        st.dataframe(df.head())

        st.markdown("### 📘 Explanation:")
        st.markdown("- This shows the first few rows of your dataset to verify structure.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Summary Statistics")
            st.write(df.describe())
        with col2:
            st.subheader("❌ Missing Values")
            missing = df.isnull().sum()
            st.write(missing[missing > 0] if missing.sum() > 0 else "✅ No missing values")

        with st.expander("🧠 Full Info"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

        st.markdown("### 💡 Insight:")
        st.markdown(f"- Your dataset contains `{df.shape[0]}` rows and `{df.shape[1]}` columns.")
        st.markdown("- Use this to identify outliers, check for nulls, and see column types.")

    # === Section 2: Univariate Analysis ===
    with st.expander("📊 2. Univariate Analysis"):
        column = st.selectbox("Select column for univariate analysis", df.columns)
        st.markdown("### 📘 Explanation:")
        st.markdown("- Visualizing single variables helps detect skewness, imbalance, or anomalies.")

        if df[column].dtype in [np.float64, np.int64]:
            fig, ax = plt.subplots()
            sns.histplot(df[column].dropna(), kde=True, ax=ax, color="teal")
            ax.set_title(f"Distribution of {column}")
            st.pyplot(fig)

            st.markdown("### 💡 Finding:")
            st.write(f"Mean: {df[column].mean():.2f}, Std: {df[column].std():.2f}, Skew: {df[column].skew():.2f}")
        else:
            fig, ax = plt.subplots()
            df[column].value_counts().plot(kind='bar', ax=ax, color="orange")
            ax.set_title(f"Value Counts of {column}")
            st.pyplot(fig)

            st.markdown("### 💡 Finding:")
            st.write(f"Top category: {df[column].mode()[0]}")

    # === Section 3: Bivariate Analysis ===
    with st.expander("🔗 3. Bivariate Analysis"):
        col1, col2 = st.columns(2)
        with col1:
            x = st.selectbox("X Variable (numerical)", num_cols, key="biv_x")
        with col2:
            y = st.selectbox("Y Variable (numerical)", [i for i in num_cols if i != x], key="biv_y")

        st.markdown("### 📘 Explanation:")
        st.markdown("- Shows how two variables relate — useful for spotting trends.")

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x, y=y, ax=ax, color="purple")
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)

        st.markdown("### 💡 Insight:")
        st.write(f"Correlation between `{x}` and `{y}`: {df[x].corr(df[y]):.2f}")

    # === Section 4: Multivariate Analysis ===
    with st.expander("📉 4. Multivariate Analysis"):
        st.markdown("### 🔥 Correlation Heatmap")
        corr = df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        st.markdown("### 📘 Explanation:")
        st.markdown("- A heatmap shows how multiple numerical features relate.")
        st.markdown("### 💡 Insight:")
        st.write("- Look for high correlations that might affect models due to multicollinearity.")

    # === Section 5: Auto Model Selection and Regression ===
    with st.expander("🧠 5. Predictive Modeling (Auto ML)"):
        target = st.selectbox("Select target column", df.columns, key="target")

        if target in num_cols:
            # Linear Regression
            st.markdown("### 🧮 Linear Regression (for numeric targets)")

            features = [col for col in num_cols if col != target]
            X = df[features].dropna()
            y = df[target].dropna()

            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.markdown(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
            st.markdown(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7, color='blue')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

            st.markdown("### 📘 Explanation:")
            st.markdown("- R² shows how well the features explain the target.")
            st.markdown("### 💡 Insight:")
            st.markdown("- Higher R² (close to 1) means better fit.")

        elif df[target].nunique() == 2:
            # Logistic Regression
            st.markdown("### 🤖 Logistic Regression (for binary targets)")

            df_clean = df.dropna(subset=[target])
            df_clean[target] = df_clean[target].astype('category').cat.codes
            features = [col for col in df_clean.columns if col != target and df_clean[col].dtype in [np.int64, np.float64]]
            X = df_clean[features]
            y = df_clean[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.markdown(f"**Accuracy:** {acc:.2f}")

            cm = confusion_matrix(y_test, y_pred)
            st.markdown("### Confusion Matrix")
            st.write(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))

            st.markdown("### 📘 Explanation:")
            st.markdown("- Logistic regression is for classifying binary outcomes.")
            st.markdown("### 💡 Insight:")
            st.markdown("- Accuracy shows correct predictions. Confusion matrix shows TP, FP, FN, TN.")

else:
    st.info("👈 Upload a CSV file to begin your analysis.")
