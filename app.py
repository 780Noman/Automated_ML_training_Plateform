import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import io
import joblib

def get_pipeline(model, X):
    """Creates a full preprocessing and modeling pipeline."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return full_pipeline

def main():
    """Defines the Streamlit application UI and logic."""
    st.set_page_config(layout="wide", page_title="Pro ML App")
    st.title("End-to-End Machine Learning Application")

    st.session_state.setdefault('pipeline', None)
    st.session_state.setdefault('features', None)
    st.session_state.setdefault('target_encoder', None)
    st.session_state.setdefault('problem_type', None)
    st.session_state.setdefault('model_name', None)
    st.session_state.setdefault('data', None)

    with st.sidebar:
        st.header("Configuration")
        st.subheader("1. Data Input")
        data_source = st.selectbox("Choose data source", ["Example", "Upload"])

        data = None
        if data_source == "Upload":
            uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'tsv'])
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'): data = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'): data = pd.read_excel(uploaded_file)
                    else: data = pd.read_csv(uploaded_file, sep='\t')
                    st.session_state.data = data
                except Exception as e: st.error(f"Error loading file: {e}")
        else:
            dataset_name = st.selectbox("Select an example dataset", ["titanic", "tips", "iris"])
            data = sns.load_dataset(dataset_name)
            st.session_state.data = data

    if st.session_state.data is not None:
        data = st.session_state.data
        
        st.header("Exploratory Data Analysis")
        st.dataframe(data.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Shape:**", data.shape)
            buffer = io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text_area("**Data Info (Types & Nulls):**", s, height=250)
        with col2:
            st.write("**Data Description:**")
            st.dataframe(data.describe())

        with st.sidebar:
            st.subheader("2. Model Configuration")
            problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression"])
            target = st.selectbox("Select Target Column", data.columns.tolist())
            
            features = [col for col in data.columns if col != target]
            st.session_state.features = features

            test_size = st.slider("Test Split Size", 0.1, 0.5, value=0.2, step=0.05)

            model_map = {
                'Regression': {'Linear Regression': LinearRegression(), 'Decision Tree': DecisionTreeRegressor(), 'Random Forest': RandomForestRegressor(), 'SVM': SVR()},
                'Classification': {'Decision Tree': DecisionTreeClassifier(), 'Random Forest': RandomForestClassifier(), 'SVM': SVC(probability=True)}
            }
            model_name = st.selectbox("Select Model", model_map[problem_type].keys())
            model = model_map[problem_type][model_name]

        if st.button("Run Analysis & Evaluate Model", type="primary"):
            with st.spinner("Training model and evaluating... Please wait."):
                try:
                    # --- CRITICAL FIX: Handle missing values in target column ---
                    data.dropna(subset=[target], inplace=True)

                    X = data[features]
                    y = data[target]
                    
                    if problem_type == 'Classification':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        st.session_state.target_encoder = le
                    
                    st.session_state.problem_type = problem_type
                    st.session_state.model_name = model_name

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    pipeline = get_pipeline(model, X_train)
                    pipeline.fit(X_train, y_train)
                    st.session_state.pipeline = pipeline
                    
                    st.header("Model Evaluation Results")
                    predictions = pipeline.predict(X_test)

                    if problem_type == 'Regression':
                        rmse = mean_squared_error(y_test, predictions, squared=False)
                        mae = mean_absolute_error(y_test, predictions)
                        r2 = r2_score(y_test, predictions)
                        st.subheader('Regression Metrics')
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("RMSE", f"{rmse:.4f}"); m_col2.metric("MAE", f"{mae:.4f}"); m_col3.metric("R²", f"{r2:.4f}")
                        
                        st.subheader("True vs. Predicted Values")
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, predictions, alpha=0.7, edgecolors='k')
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                        ax.set_xlabel('True Values'); ax.set_ylabel('Predicted Values'); ax.set_title('True vs Predicted Values')
                        st.pyplot(fig)
                    else:
                        acc = accuracy_score(y_test, predictions)
                        prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
                        rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
                        st.subheader('Classification Metrics')
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        m_col1.metric("Accuracy", f"{acc:.4f}"); m_col2.metric("Precision", f"{prec:.4f}"); m_col3.metric("Recall", f"{rec:.4f}"); m_col4.metric("F1 Score", f"{f1:.4f}")

                        st.subheader("Confusion Matrix")
                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, cmap=plt.cm.Blues)
                        st.pyplot(fig)

                    st.success("Evaluation complete!")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.exception(e)

    if st.session_state.pipeline is not None:
        st.header("Live Prediction")
        with st.form(key='prediction_form'):
            st.write("Input new data to get a prediction from the trained model.")
            inputs = {}
            for feature in st.session_state.features:
                if data[feature].dtype == np.number:
                    inputs[feature] = st.number_input(f"Input for {feature}", value=float(data[feature].mean()))
                else:
                    inputs[feature] = st.selectbox(f"Input for {feature}", options=data[feature].unique())
            
            submit_button = st.form_submit_button(label='Predict')

            if submit_button:
                input_df = pd.DataFrame([inputs])
                prediction = st.session_state.pipeline.predict(input_df)
                
                if st.session_state.problem_type == 'Classification' and st.session_state.target_encoder:
                    final_prediction = st.session_state.target_encoder.inverse_transform(prediction)[0]
                    st.success(f"The predicted class is: **{final_prediction}**")
                else:
                    st.success(f"The predicted value is: **{prediction[0]:.4f}**")
        
        st.header("Download Model")
        buffer = io.BytesIO()
        joblib.dump(st.session_state.pipeline, buffer)
        st.download_button(
            label="Download Trained Pipeline",
            data=buffer,
            file_name=f"{st.session_state.model_name}_pipeline.joblib",
            mime="application/octet-stream"
        )

if __name__ == "__main__":
    main()
