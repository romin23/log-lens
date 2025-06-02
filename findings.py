import pandas as pd

# Define the data as a list of dictionaries
data = [
    {
        "Model": "Logistic Regression",
        "Accuracy": 1.000,
        "Precision": 1.000,
        "Recall": 1.000,
        "F1 Score": 1.000,
        "Training Time": "Very Fast",
        "Prediction Time": "Very Fast",
        "Memory Usage": "Low",
        "Notes": "Excellent performance with minimal resource consumption."
    },
    {
        "Model": "K-Nearest Neighbors (KNN)",
        "Accuracy": 1.000,
        "Precision": 1.000,
        "Recall": 1.000,
        "F1 Score": 1.000,
        "Training Time": "Negligible",
        "Prediction Time": "Slow",
        "Memory Usage": "High",
        "Notes": "Fast training but slow prediction due to distance calculations."
    },
    {
        "Model": "Support Vector Machine (SVM)",
        "Accuracy": 1.000,
        "Precision": 1.000,
        "Recall": 1.000,
        "F1 Score": 1.000,
        "Training Time": "Moderate",
        "Prediction Time": "Moderate",
        "Memory Usage": "High",
        "Notes": "High accuracy but resource-intensive; may require parameter tuning."
    },
    {
        "Model": "Random Forest",
        "Accuracy": 1.000,
        "Precision": 1.000,
        "Recall": 1.000,
        "F1 Score": 1.000,
        "Training Time": "Moderate",
        "Prediction Time": "Moderate",
        "Memory Usage": "High",
        "Notes": "Robust performance; handles overfitting well but consumes more memory."
    },
    {
        "Model": "Linear Discriminant Analysis (LDA)",
        "Accuracy": 1.000,
        "Precision": 1.000,
        "Recall": 1.000,
        "F1 Score": 1.000,
        "Training Time": "Very Fast",
        "Prediction Time": "Very Fast",
        "Memory Usage": "Low",
        "Notes": "Efficient and interpretable; assumes linear separability."
    },
    {
        "Model": "Neural Network",
        "Accuracy": 1.000,
        "Precision": 1.000,
        "Recall": 1.000,
        "F1 Score": 1.000,
        "Training Time": "Slow",
        "Prediction Time": "Fast",
        "Memory Usage": "High",
        "Notes": "High accuracy; requires more training time and resources."
    },
    {
        "Model": "Naive Bayes",
        "Accuracy": 0.967,
        "Precision": 0.976,
        "Recall": 0.967,
        "F1 Score": 0.969,
        "Training Time": "Very Fast",
        "Prediction Time": "Very Fast",
        "Memory Usage": "Very Low",
        "Notes": "Simple and fast; performs well with large datasets."
    },
    {
        "Model": "AdaBoost",
        "Accuracy": 0.962,
        "Precision": 0.964,
        "Recall": 0.962,
        "F1 Score": 0.963,
        "Training Time": "Moderate",
        "Prediction Time": "Fast",
        "Memory Usage": "Moderate",
        "Notes": "Boosting improves accuracy; sensitive to noisy data."
    },
    {
        "Model": "Decision Tree",
        "Accuracy": 0.929,
        "Precision": 0.936,
        "Recall": 0.929,
        "F1 Score": 0.931,
        "Training Time": "Very Fast",
        "Prediction Time": "Very Fast",
        "Memory Usage": "Low",
        "Notes": "Easy to interpret; prone to overfitting without pruning."
    },
    {
        "Model": "Quadratic Discriminant Analysis (QDA)",
        "Accuracy": 0.465,
        "Precision": 0.216,
        "Recall": 0.465,
        "F1 Score": 0.295,
        "Training Time": "Fast",
        "Prediction Time": "Fast",
        "Memory Usage": "Moderate",
        "Notes": "Poor performance; issues with covariance matrix estimation."
    }
]

def get():
    df = pd.DataFrame(data)
    return df
