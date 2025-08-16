# End-to-End AutoML Web Application

A comprehensive and user-friendly AutoML web application built with Streamlit. This no-code tool empowers users to upload a dataset, perform exploratory data analysis, and automatically train, evaluate, and test multiple machine learning models for both classification and regression tasks.

## ✨ Features

- **Data Input Options**: Upload your own datasets (`.csv`, `.xlsx`, `.tsv`) or use classic example datasets (`titanic`, `tips`, `iris`) provided in the app.
- **Automated Exploratory Data Analysis (EDA)**: Instantly view key information about your dataset:
  - Data preview (first 5 rows)
  - Dataset shape (rows, columns)
  - Column data types and non-null counts
  - Detailed statistical summary for numerical columns
- **Flexible Model Configuration**:
  - Supports both **Classification** and **Regression** tasks.
  - Dynamically select the target variable and features.
  - Adjust the training/test data split ratio with a slider.
- **Multiple Model Support**: Choose from a variety of popular `scikit-learn` models:
  - **Regression**: Linear Regression, Decision Tree, Random Forest, Support Vector Regressor (SVR).
  - **Classification**: Decision Tree, Random Forest, Support Vector Classifier (SVC).
- **Automated Preprocessing**: A robust `scikit-learn` pipeline handles common preprocessing steps automatically:
  - **Numerical Features**: Median value imputation and standard scaling.
  - **Categorical Features**: Most frequent value imputation and one-hot encoding.
- **Comprehensive Model Evaluation**:
  - **Regression Metrics**: R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
  - **Classification Metrics**: Accuracy, Precision, Recall, and F1-Score.
  - **Visualizations**:
    - **Regression**: A scatter plot comparing true values against predicted values.
    - **Classification**: A detailed confusion matrix.
- **Live Prediction Engine**: After training, use an interactive form to input new data and receive instant predictions from the model.
- **Download Trained Model**: Download the entire trained pipeline (preprocessor + model) as a `.joblib` file for easy deployment or offline use.

## 🚀 Tech Stack

- **Core Framework**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/)
- **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Model Persistence**: [Joblib](https://joblib.readthedocs.io/)

## ⚙️ Setup and Local Installation

Follow these steps to get the application running on your local machine.

**1. Clone the Repository**

```bash
git clone https://github.com/780Noman/Automated_ML_training_Plateform.git
cd YOUR_REPOSITORY
```

> **Note**: Replace `YOUR_USERNAME/YOUR_REPOSITORY` with your actual GitHub username and repository name.

**2. Create a Virtual Environment (Recommended)**

- On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

- On Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit Application**

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 📖 How to Use the App

1. **Choose Data Source**: In the sidebar, select either "Upload" to use your own file or "Example" to use a built-in dataset.
2. **Configure Model**:
   - Select the **Problem Type** (Regression or Classification).
   - Choose the **Target Column** you want to predict.
   - Adjust the **Test Split Size** using the slider.
   - Select the **Machine Learning Model** to train.
3. **Run Analysis**: Click the **"Run Analysis & Evaluate Model"** button to start the training and evaluation process.
4. **Review Results**: Examine the performance metrics and visualizations displayed in the main panel.
5. **Make Live Predictions**: Scroll down to the "Live Prediction" section, fill in the input fields, and click "Predict".
6. **Download Model**: Click the **"Download Trained Pipeline"** button to save the model for future use.
