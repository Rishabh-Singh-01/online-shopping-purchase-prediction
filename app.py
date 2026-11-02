import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Online Shopping Purchase Prediction", page_icon="🛒", layout="wide"
)

st.title("Online Shopping Purchase Prediction")
st.write(
    "Evaluate five machine learning classification models "
    "on online shopping session data."
)


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

FEATURE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

TARGET_COLUMN = "Revenue"


# --------------------------------------------------
# 3. LOAD MODEL
# --------------------------------------------------


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# 4. UPLOAD DATASET
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a test CSV file to begin.")

else:
    try:
        test_df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(test_df.head())

        st.write(f"Dataset shape: {test_df.shape}")

        expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

        missing_columns = [
            col for col in expected_columns if col not in test_df.columns
        ]

        extra_columns = [col for col in test_df.columns if col not in expected_columns]

        # Dataset validation
        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")

        elif extra_columns:
            st.error(f"Unexpected columns: {extra_columns}")

        elif test_df.isnull().any().any():
            st.error("The dataset contains missing values.")

        elif not test_df[TARGET_COLUMN].isin([True, False]).all():
            st.error("Revenue must contain only True or False values.")

        else:
            st.success("Dataset validation successful!")

            # --------------------------------------------------
            # 5. MODEL SELECTION
            # --------------------------------------------------

            st.header("2. Select Classification Model")

            selected_model_name = st.selectbox(
                "Choose a classification model", list(MODEL_FILES.keys())
            )

            model_path = os.path.join(MODEL_DIR, MODEL_FILES[selected_model_name])

            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")

            else:
                selected_model = load_model(model_path)

                st.success(f"{selected_model_name} loaded successfully.")

                # --------------------------------------------------
                # 6. MODEL EVALUATION
                # --------------------------------------------------

                st.header("3. Model Evaluation")

                X_test = test_df[FEATURE_COLUMNS]
                y_test = test_df[TARGET_COLUMN]

                y_pred = selected_model.predict(X_test)
                y_prob = selected_model.predict_proba(X_test)[:, 1]

                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "AUC": roc_auc_score(y_test, y_prob),
                    "Precision": precision_score(
                        y_test, y_pred, pos_label=True, zero_division=0
                    ),
                    "Recall": recall_score(
                        y_test, y_pred, pos_label=True, zero_division=0
                    ),
                    "F1 Score": f1_score(
                        y_test, y_pred, pos_label=True, zero_division=0
                    ),
                    "MCC": matthews_corrcoef(y_test, y_pred),
                }

                # Display metrics
                st.subheader("Evaluation Metrics")

                metric_columns = st.columns(3)

                for index, (metric_name, metric_value) in enumerate(metrics.items()):
                    with metric_columns[index % 3]:
                        st.metric(label=metric_name, value=f"{metric_value:.4f}")

                # --------------------------------------------------
                # 7. CONFUSION MATRIX
                # --------------------------------------------------

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(y_test, y_pred, labels=[False, True])

                fig, ax = plt.subplots(figsize=(6, 4))

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["No Purchase", "Purchase"],
                    yticklabels=["No Purchase", "Purchase"],
                    ax=ax,
                )

                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")
                ax.set_title(f"{selected_model_name} - Confusion Matrix")

                st.pyplot(fig)
                plt.close(fig)

                # --------------------------------------------------
                # 8. CLASSIFICATION REPORT
                # --------------------------------------------------

                st.subheader("Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    labels=[False, True],
                    target_names=["No Purchase", "Purchase"],
                    output_dict=True,
                    zero_division=0,
                )

                report_df = pd.DataFrame(report).transpose()

                st.dataframe(report_df.round(4), use_container_width=True)

                # --------------------------------------------------
                # 9. ALL-MODEL COMPARISON
                # --------------------------------------------------

                st.header("4. Compare All Classification Models")

                comparison_results = []

                for model_name, filename in MODEL_FILES.items():
                    comparison_path = os.path.join(MODEL_DIR, filename)

                    comparison_model = load_model(comparison_path)

                    comparison_pred = comparison_model.predict(X_test)

                    comparison_prob = comparison_model.predict_proba(X_test)[:, 1]

                    comparison_results.append(
                        {
                            "Model": model_name,
                            "Accuracy": accuracy_score(y_test, comparison_pred),
                            "AUC": roc_auc_score(y_test, comparison_prob),
                            "Precision": precision_score(
                                y_test, comparison_pred, pos_label=True, zero_division=0
                            ),
                            "Recall": recall_score(
                                y_test, comparison_pred, pos_label=True, zero_division=0
                            ),
                            "F1": f1_score(
                                y_test, comparison_pred, pos_label=True, zero_division=0
                            ),
                            "MCC": matthews_corrcoef(y_test, comparison_pred),
                        }
                    )

                comparison_df = pd.DataFrame(comparison_results)

                st.dataframe(
                    comparison_df.set_index("Model").round(4), use_container_width=True
                )

                # --------------------------------------------------
                # 10. PERFORMANCE VISUALIZATION
                # --------------------------------------------------

                st.subheader("Model Performance Comparison")

                selected_metrics = st.multiselect(
                    "Select metrics to compare",
                    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                    default=["F1", "MCC"],
                )

                if selected_metrics:
                    chart_data = comparison_df.set_index("Model")[selected_metrics]

                    st.bar_chart(chart_data)

                else:
                    st.info("Select at least one metric to display the chart.")

    except Exception as error:
        st.error(f"An error occurred: {error}")
