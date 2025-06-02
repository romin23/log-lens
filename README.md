
# 🔍 Project Overview: Hybrid Log Classification System

In this project, we developed a comprehensive log classification system that synergizes multiple machine learning approaches to effectively handle diverse and complex log data. By integrating rule-based methods with data-driven models, our system adeptly processes structured, semi-structured, and unstructured logs. This hybrid approach ensures robust performance across varying data complexities, enhancing error detection and system monitoring capabilities.

---

## Project Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://romin-log-lens.streamlit.app)


---

## 🧪 1. Generating Synthetic Data

To simulate real-world scenarios without compromising privacy, we utilized synthetic data generation techniques. Tools like `ydata-synthetic` offer robust solutions for creating realistic synthetic datasets, facilitating experimentation and model development without the need for sensitive information.

---

## 🧹 2. Data Cleaning and Exploratory Data Analysis (EDA)

Before diving into modeling, we performed thorough data cleaning and EDA to understand the dataset's structure and nuances.

* **Handling Missing Values**: Identified and addressed any missing entries to ensure data integrity.

* **Encoding Categorical Variables**: Transformed categorical features into numerical representations suitable for machine learning algorithms.

* **Feature Scaling**: Applied `StandardScaler` or `MinMaxScaler` where appropriate, particularly for algorithms sensitive to feature scales.

* **Visualization**: Utilized histograms, box plots, and correlation matrices to uncover patterns and relationships within the data.

---

## 🤖 3. Model Training and Evaluation

We experimented with various classification models to identify the most effective algorithm for our dataset.

### Models Tested:

* **Logistic Regression**

* **K-Nearest Neighbors (KNN)**

* **Support Vector Machine (SVM)**

* **Decision Tree**

* **Random Forest**

* **AdaBoost**

* **Naive Bayes**

* **Neural Network**

* **Linear Discriminant Analysis (LDA)**

* **Quadratic Discriminant Analysis (QDA)**

### Evaluation Metrics:

* **Accuracy**: Proportion of correctly predicted instances.

* **Precision**: Measure of exactness or quality of positive predictions.

* **Recall**: Measure of completeness or quantity of positive predictions.

* **F1 Score**: Harmonic mean of precision and recall, providing a balance between the two.

### Findings:

![findings](resources/findings.png)

**Recommendation**: Considering both performance and computational efficiency, **Logistic Regression** emerged as the optimal choice, delivering perfect accuracy with minimal resource consumption.

---

## 🌐 4. Deploying the Model with Streamlit

To make our model accessible and interactive, we developed a web application using Streamlit.

### Key Features:

* **User-Friendly Interface**: Intuitive design allowing users to navigate and utilize the app effortlessly.

* **Model Selection**: Users can choose from the trained models to make predictions.

* **Data Upload**: Option to upload CSV files for batch predictions.

* **Real-Time Predictions**: Immediate output of predictions upon data input.

* **Downloadable Results**: Ability to download prediction results as a CSV file.

### How It Works:

1. **Upload Data**: Users upload their dataset in CSV format.

2. **Select Model**: Choose the preferred machine learning model from the available options.

3. **Make Predictions**: The app processes the data and outputs predictions.

4. **Download Results**: Users can download the predictions for further analysis or reporting.

---

## 🔮 5. Future Enhancements

To further improve the application, we plan to implement the following features:

* **Model Explanation**: Integrate tools like SHAP or LIME to provide insights into model predictions.

* **Model Retraining**: Allow users to retrain models with their own data directly through the app.

* **Enhanced Visualizations**: Incorporate advanced charts and graphs for better data interpretation.

---

By following this end-to-end approach, we've created a robust and user-friendly machine learning application that streamlines the process from data ingestion to prediction deployment. Whether you're a data enthusiast or a seasoned professional, this project serves as a valuable blueprint for developing and deploying machine learning solutions.

Feel free to customize and expand upon this framework to suit your specific needs and objectives!