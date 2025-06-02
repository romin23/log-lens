import streamlit as st
import pandas as pd
import findings
from log_pred import classify_with_bert

def main():
    st.set_page_config(
        page_title="Log-Lens",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Welcome to Log-Lens🔍")
    st.write("A hybrid log classification system that integrates multiple machine learning approaches to effectively process and categorize diverse log patterns")
    st.markdown("Classes modelled by Log-Lens🔍: :violet-badge[HTTP Status] " \
    "            :orange-badge[Critical Error] " \
    "            :green-badge[Resource Usage]"
    "            :red-badge[Error] " \
    "            :blue-badge[System Notification] " \
    "            :violet-badge[Security Alert]")
    st.sidebar.title("Log-Lens Navigation🔍")
    
    st.expander("🔑 Key Findings", expanded=False).dataframe(findings.get(), use_container_width=True)
    # user_input = st.text_input("Enter some text:")
    
    tab1, tab2 = st.tabs(["🤖 Model Prediction", "📒 Model Training Details"])

    with tab2:
        st.markdown("""
        # 🧠 Model Training & Data Preparation

        ---

        ## 📋 Overview

        The goal is to classify system log messages into one of the following categories:
        - `HTTP Status`
        - `Security Alert`
        - `System Notification`
        - `Error`
        - `Resource Usage`
        - `Critical Error`

        Log messages are highly imbalanced in distribution, which was the key challenge addressed during model development.

        ---

        ## 🧹 Step 1: Data Preprocessing

        ### 1.1. Sentence Embedding

        Log messages (free-text) were transformed into fixed-length numerical vectors using:

        ```python
        from sentence_transformers import SentenceTransformer

        model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
        X = model_embedding.encode(df['log_message'].tolist())
        ```

        This model converts each log message into a dense 384-dimensional embedding, capturing semantic meaning suitable for downstream classification.

        ### 1.2. Label Encoding

        Target labels were converted into a numpy array for modeling:

        ```python
        y = df['target_label'].values
        ```

        ---

        ## ⚖️ Step 2: Handling Class Imbalance

        Class imbalance was significant, with some classes having <200 examples and others over 1000. To handle this:

        ### ✅ Option 1: **SMOTE + Custom Sampling**

        ```python
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(sampling_strategy={
            'Critical Error': 400,
            'Resource Usage': 400,
            'Error': 400,
            'System Notification': 400,
            'Security Alert': 400
        }, random_state=42)

        X_resampled, y_resampled = smote.fit_resample(X, y)
        ```

        We explicitly avoided oversampling the majority class (`HTTP Status` with 1017 samples), keeping it unchanged to avoid excessive duplication. This approach ensures class distribution is more uniform without artificially bloating the dataset.

        ### ✅ Option 2: **Balanced Random Forest (Built-in Resampling)**

        ```python
        from imblearn.ensemble import BalancedRandomForestClassifier
        ```

        This classifier automatically balances classes by undersampling the majority class in each bootstrap sample **per tree**, eliminating the need for explicit resampling. It's highly effective for imbalanced datasets.

        ---

        ## ⚙️ Step 3: Feature Scaling

        Even though tree-based models don’t require scaling, we used `StandardScaler` to normalize features (embeddings) to zero mean and unit variance, especially for models sensitive to feature scale:

        ```python
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        ```

        ---

        ## 🏋️ Step 4: Model Training

        ### 🔹 Logistic Regression (Baseline)

        ```python
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        clf.fit(X_train_scaled, y_train)
        ```

        ### 🔹 K-Nearest Neighbors (KNN)

        ```python
        from sklearn.neighbors import KNeighborsClassifier

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train_scaled, y_train)
        ```

        ### 🔹 Balanced Random Forest (Preferred for Imbalance)

        ```python
        from imblearn.ensemble import BalancedRandomForestClassifier

        clf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        ```

        ---

        ## 🧪 Step 5: Evaluation

        ```python
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

        y_pred = clf.predict(X_test_scaled)
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred, normalize='true')
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_).plot()
        ```

        ---

        ## 💾 Step 6: Saving Models & Scalers

        ```python
        import joblib

        joblib.dump(clf, 'models/balanced_rf.joblib')
        joblib.dump(scaler, 'models/scaler.joblib')
        ```

        ---

        ## ✅ Summary

        | Component | Technique |
        |----------|------------|
        | Text Embedding | Sentence-BERT (`all-MiniLM-L6-v2`) |
        | Imbalance Handling | SMOTE (explicit) or Balanced RF (inherent) |
        | Feature Scaling | StandardScaler |
        | Models Trained | Logistic Regression, KNN, Balanced Random Forest |
        | Evaluation | F1-score, confusion matrix |

        This approach ensures robust classification while mitigating class imbalance, improving generalization across all log categories.
        """, unsafe_allow_html=True)


    # Data Profiling
    with st.sidebar.expander("📌 Introduction", expanded=False):
        st.markdown("""
    This application is designed to classify system and application log messages into meaningful categories using machine learning models. It supports intelligent classification of logs such as `Error`, `HTTP Status`, `Security Alert`, and more, helping users quickly analyze large volumes of logs. Users can upload their own CSV files, choose from multiple trained models, and download the results with predicted labels. The app handles class imbalance, uses semantic embeddings, and provides end-to-end log intelligence at your fingertips. 🔍📊
    """)
    
    with st.sidebar.expander(" 👨‍💻 Model Selection", expanded=False):
        models = [" ","Logistic Regression", "K-Nearest Neighbors", "Random Forest", "Support Vector Machine", "Neural Network", "AdaBoost", "Decision Tree", "Quadratic Discriminant Analysis (QDA)", "Linear Discriminant Analysis (LDA)"]
        st.write("Select a model to classify the logs:")
        model = st.selectbox("Select Model", models)
        if model != " ":
            options = ["Use Sample Data", "Test your own data"]
            selection = st.segmented_control("Test Data", options, selection_mode="single")
        if model != " " and selection == "Use Sample Data":
            sample_data = st.selectbox("Select Sample Data", [None,"Synthetic Test Logs", "Training Test Logs", "Real-world Test Logs"], key="sample_data_selector")
            if sample_data is not None:
                with tab1:
                    if sample_data == "Synthetic Test Logs":
                        up_dataframe = pd.read_csv('resources/test_logs.csv')
                    elif sample_data == "Training Test Logs":
                        up_dataframe = pd.read_csv('resources/output.csv')
                    elif sample_data == "Real-world Test Logs":
                        up_dataframe = pd.read_csv('resources/test.csv')
                    st.success("Sample data loaded successfully!")
                    st.dataframe(up_dataframe.head(), use_container_width=True)
                    if st.button("Predict"):
                        st.write(f"Classifying logs using the {model} model...")
                        classified_df = classify_with_bert(up_dataframe, 'log_message', model)
                        st.write("Classified Data:")
                        st.dataframe(classified_df.head(), use_container_width=True)
                        st.download_button(
                            label="Download Classified Logs",
                            data=classified_df.to_csv(index=False).encode('utf-8'),
                            file_name='classified_logs.csv',
                            mime='text/csv'
                        )
    
    
        
    with tab1:  
        if model != " " and selection == "Test your own data":
            up_data = st.file_uploader("Upload your log file here", type=["csv"], key="log_file_uploader")
            st.caption("Upload your log file to classify it using the selected model")

            if up_data is not None:
                up_dataframe = pd.read_csv(up_data)

                st.success("File uploaded successfully!")
                st.dataframe(up_dataframe.head(), use_container_width=True)
                st.text_input("Enter the column name containing log messages:", value="log_message", key="log_column_input")
                log_column = st.session_state.get("log_column_input", "log_message")

                if st.button("Classify Logs"):
                    st.write("Classifying logs using the selected model...")
                    if log_column in up_dataframe.columns:
                        # with st.spinner("Classifying logs..."):
                        # Call the classification function
                        classified_df = classify_with_bert(up_dataframe, log_column, model)
                        st.write("Classified Data:")
                        st.dataframe(classified_df.head(), use_container_width=True)
                        st.download_button(
                            label="Download Classified Logs",
                            data=classified_df.to_csv(index=False).encode('utf-8'),
                            file_name='classified_logs.csv',
                            mime='text/csv'
                        )
                    else:
                        st.error(f"Column '{log_column}' not found in the uploaded file.")
                    # Here you would typically call a function to process the uploaded file and apply the selected model
                    # For example: findings.process_file(up_data, model)
            else:
                st.write("Please upload a log file to proceed.")
if __name__ == "__main__":
    main()