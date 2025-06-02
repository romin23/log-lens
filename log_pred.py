import joblib
import streamlit as st
from sentence_transformers import SentenceTransformer

model_embedding = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
scaler = joblib.load("models/logistic_scaler.joblib")

def classify_with_bert(df, col_name, model):

    with st.spinner("Loading Model..."):
        if model == "Logistic Regression":
            model_classification = joblib.load("models/logistic.joblib")
        elif model == "K-Nearest Neighbors":
            model_classification = joblib.load("models/knn.joblib")  
        elif model == "Random Forest":
            model_classification = joblib.load("models/randomForest.joblib")
        elif model == "Support Vector Machine":
            model_classification = joblib.load("models/svm.joblib")
        elif model == "Neural Network":
            model_classification = joblib.load("models/mlp.joblib") 
        elif model == "AdaBoost":
            model_classification = joblib.load("models/ABoost.joblib")
        elif model == "Decision Tree":
            model_classification = joblib.load("models/dtree.joblib")
        elif model == "Quadratic Discriminant Analysis (QDA)":
            model_classification = joblib.load("models/QuadraticDiscriminantAnalysis.joblib")
        elif model == "Linear Discriminant Analysis (LDA)":
            model_classification = joblib.load("models/LinearDiscriminantAnalysis.joblib")
    
    with st.spinner("Generating Embeddings..."):
        # Encode the log messages using BERT embeddings
        # embeddings= scaler.transform(model_embedding.encode(df[col_name].tolist())
        embeddings = model_embedding.encode(df[col_name].tolist())
        embeddings_scaled = scaler.transform(embeddings)

    with st.spinner("Classifying Logs..."):
    # Predict probabilities for each class
        probabilities = model_classification.predict_proba(embeddings_scaled)
    
        # Determine the predicted labels with a threshold
        predicted_labels = []
        for prob in probabilities:
            if max(prob) < 0.5:
                predicted_labels.append("Unclassified")
            else:
                predicted_labels.append(model_classification.classes_[prob.argmax()])
    
        # Add the predictions to the DataFrame
        df = df.copy()  # To avoid modifying the original DataFrame
        df['prediction'] = predicted_labels
        return df