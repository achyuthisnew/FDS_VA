# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import json

# Load datasets
raw_data_path = "https://raw.githubusercontent.com/achyuthisnew/FDS_VA/refs/heads/main/Product_Purchase_Prediction_Synthetic%20(1).csv"
preprocessed_data_path = "https://raw.githubusercontent.com/achyuthisnew/FDS_VA/refs/heads/main/Preprocessed_Product_Purchase_Prediction%20(1).csv"

# Read raw and preprocessed data
try:
    df_raw = pd.read_csv(raw_data_path)
    df_preprocessed = pd.read_csv(preprocessed_data_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title and Sidebar
st.title("Interactive Dashboard for Product Purchase Prediction")
st.sidebar.title("Options")

# Section 1: Data Distribution Visualization
st.header("1. Data Distribution Visualization")

data_selection = st.sidebar.radio("Choose data to display:", ("Raw Data", "Preprocessed Data"))
selected_data = df_raw if data_selection == "Raw Data" else df_preprocessed

st.write(f"### {data_selection} Distribution")
for col in selected_data.select_dtypes(include=["object", "int64", "float64"]).columns:
    fig = px.histogram(selected_data, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Section 2: Model Performance Metrics
st.header("2. Model Performance Metrics")

# Classification report for raw data
raw_report_json = """
{
    "0": {"precision": 0.65, "recall": 0.82, "f1-score": 0.73, "support": 130},
    "1": {"precision": 0.42, "recall": 0.23, "f1-score": 0.30, "support": 74},
    "accuracy": 0.6078,
    "macro avg": {"precision": 0.54, "recall": 0.53, "f1-score": 0.51, "support": 204},
    "weighted avg": {"precision": 0.57, "recall": 0.61, "f1-score": 0.57, "support": 204}
}
"""

# Classification report for preprocessed data
preprocessed_report_json = """
{
    "0": {"precision": 0.67, "recall": 0.92, "f1-score": 0.78, "support": 130},
    "1": {"precision": 0.62, "recall": 0.22, "f1-score": 0.32, "support": 74},
    "accuracy": 0.6667,
    "macro avg": {"precision": 0.64, "recall": 0.57, "f1-score": 0.55, "support": 204},
    "weighted avg": {"precision": 0.65, "recall": 0.67, "f1-score": 0.61, "support": 204}
}
"""

# Load JSON reports from strings
try:
    raw_report = json.loads(raw_report_json)
    preprocessed_report = json.loads(preprocessed_report_json)
except Exception as e:
    st.error(f"Error parsing classification reports: {e}")
    st.stop()

# Display metrics for raw data
st.write("### Training Metrics (Raw Data)")
raw_metrics = ["precision", "recall", "f1-score"]
raw_metric_data = {metric: [raw_report[str(i)][metric] for i in range(2)] for metric in raw_metrics}
raw_metric_data["Class"] = ["Class 0", "Class 1"]

raw_df_metrics = pd.DataFrame(raw_metric_data)
fig = px.bar(raw_df_metrics, x="Class", y=raw_metrics, barmode="group", title="Classification Metrics for Raw Data")
st.plotly_chart(fig, use_container_width=True)

# Display metrics for preprocessed data
st.write("### Training Metrics (Preprocessed Data)")
preprocessed_metrics = ["precision", "recall", "f1-score"]
preprocessed_metric_data = {metric: [preprocessed_report[str(i)][metric] for i in range(2)] for metric in preprocessed_metrics}
preprocessed_metric_data["Class"] = ["Class 0", "Class 1"]

preprocessed_df_metrics = pd.DataFrame(preprocessed_metric_data)
fig = px.bar(preprocessed_df_metrics, x="Class", y=preprocessed_metrics, barmode="group", title="Classification Metrics for Preprocessed Data")
st.plotly_chart(fig, use_container_width=True)

# Section 3: Model Comparison
st.header("3. Model Comparison")

# Model comparison table
comparison_data = {
    "Model": ["Raw Data", "Preprocessed Data"],
    "Accuracy": [0.6078, 0.6667],
    "Precision": [0.42, 0.62],
    "Recall": [0.23, 0.22],
    "F1-Score": [0.30, 0.32],
    "ROC-AUC": [0.53, 0.57]
}
df_comparison = pd.DataFrame(comparison_data)

# Visualization of model comparison
fig = px.bar(df_comparison, x="Model", y=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
             barmode="group", title="Performance Comparison Between Raw and Preprocessed Data")
st.plotly_chart(fig, use_container_width=True)

# Display comparison table
st.write("### Comparison Table")
st.dataframe(df_comparison)

# Section 4: Insights
st.header("4. Insights")
st.markdown("""
- **Raw Data**: The model achieves an accuracy of 60.78%, with poor recall for Class 1, indicating difficulty in identifying positive cases.
- **Preprocessed Data**: After preprocessing, the model's accuracy improves to 66.67%. The precision for Class 1 also improves, but recall remains low.
- **Significance**: Preprocessing enhances the model's overall performance, reflected in a higher accuracy and better weighted-average metrics.
- Use the charts and tables above to analyze and compare performance metrics interactively.
""")
