# Online Shopping Purchase Intention Prediction

## 1. Problem Statement

The objective of this project is to develop and evaluate machine learning classification models that predict whether an online shopping session will result in a purchase.

The project uses the Online Shoppers Purchasing Intention dataset to identify patterns in user browsing behaviour and determine the likelihood of a purchase (`Revenue = True`).

Five classification algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using six performance metrics: Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is also developed to allow users to upload test data, select a trained model, and view its evaluation results.

## 2. Dataset Description

The project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

- **Dataset source:** [UCI Machine Learning Repository – Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

- **Number of instances:** 12,330
- **Number of input features:** 17
- **Target variable:** `Revenue`
- **Problem type:** Binary classification
- **Missing values:** None

The target variable indicates whether a shopping session resulted in a purchase:

- `True`: The visitor completed a purchase.
- `False`: The visitor did not complete a purchase.

The dataset contains numerical, categorical, and binary features.

**Numerical features:**

- Administrative
- Administrative_Duration
- Informational
- Informational_Duration
- ProductRelated
- ProductRelated_Duration
- BounceRates
- ExitRates
- PageValues
- SpecialDay

**Categorical features:**

- Month
- OperatingSystems
- Browser
- Region
- TrafficType
- VisitorType

**Binary feature:**

- Weekend

**Target feature:**

- Revenue

The dataset has an imbalanced target distribution:

| Revenue   | Number of instances | Percentage |
| --------- | ------------------: | ---------: |
| False     |              10,422 |     84.53% |
| True      |               1,908 |     15.47% |
| **Total** |          **12,330** |   **100%** |

Since the majority of sessions do not result in a purchase, accuracy alone may not provide a complete picture of model performance. Therefore, additional classification metrics are used for evaluation.

## 3. GitHub Repository Link

**GitHub Repository:** https://github.com/Rishabh-Singh-01/online-shopping-purchase-prediction

The repository contains the model training notebook, Streamlit application source code, trained model files, test dataset, evaluation results, and dependency information required to reproduce or run the project.

## 4. Models Used

Five classification algorithms were trained and evaluated on the same held-out test dataset.

### Evaluation Metrics

The following six metrics are used to compare model performance:

| Metric    | Description                                                                                                                            |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Accuracy  | Proportion of correctly classified instances among all instances.                                                                      |
| AUC       | Area Under the ROC Curve; measures the model's ability to distinguish between the two classes across classification thresholds.        |
| Precision | Proportion of predicted purchases that were actual purchases.                                                                          |
| Recall    | Proportion of actual purchases correctly identified by the model.                                                                      |
| F1-score  | Harmonic mean of Precision and Recall.                                                                                                 |
| MCC       | Matthews Correlation Coefficient; measures the quality of binary classifications while considering all four confusion matrix outcomes. |

### Model Performance Comparison

The following table presents the evaluation results for all five models on the test dataset.

| ML Model Name        | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
| -------------------- | -------: | -----: | --------: | -----: | -----: | -----: |
| Logistic Regression  |   0.8812 | 0.8830 |    0.7405 | 0.3586 | 0.4832 | 0.4610 |
| Decision Tree        |   0.8646 | 0.7435 |    0.5622 | 0.5681 | 0.5651 | 0.4849 |
| KNN                  |   0.8715 | 0.7572 |    0.6419 | 0.3848 | 0.4812 | 0.4307 |
| Gaussian Naive Bayes |   0.2733 | 0.7330 |    0.1727 | 0.9738 | 0.2934 | 0.1292 |
| SVM                  |   0.8873 | 0.8817 |    0.7063 | 0.4660 | 0.5615 | 0.5142 |

All models were evaluated using the same training-test split to ensure a consistent comparison.

## 5. Model Performance Observations

### Observations for Each Model

| Model                  | Observations                                                                                                                                                                                                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression    | Achieved the highest AUC (0.8830) and Precision (0.7405). It performs well in distinguishing between purchasers and non-purchasers and produces fewer false positives. However, its Recall (0.3586) is relatively low, meaning it misses a considerable number of actual purchases. |
| Decision Tree          | Achieved the highest F1-score (0.5651) among the five models. It also has the highest Recall (0.5681) among the models other than Gaussian Naive Bayes. It provides a more balanced Precision-Recall trade-off than Logistic Regression and KNN.                                    |
| K-Nearest Neighbors    | Achieved moderate Accuracy (0.8715) and Precision (0.6419), but its Recall (0.3848) and F1-score (0.4812) are relatively low. It does not achieve the best result in any of the six evaluation metrics.                                                                             |
| Gaussian Naive Bayes   | Achieved very high Recall (0.9738), identifying nearly all actual purchases. However, its Precision (0.1727) and Accuracy (0.2733) are very low. It generates a large number of false positives, incorrectly classifying many non-purchasing sessions as purchases.                 |
| Support Vector Machine | Achieved the highest Accuracy (0.8873) and MCC (0.5142). It provides the strongest overall balance among the evaluated models, with relatively good Precision, Recall, and F1-score.                                                                                                |

### Overall Winner

Based on the evaluation results, **Support Vector Machine (SVM) is considered the overall best-performing model** for this dataset.

It achieved the highest Accuracy (0.8873) and MCC (0.5142), indicating the strongest overall classification performance among the five models based on these metrics.

However, the choice of the best model depends on the application requirements:

- **SVM:** Best overall balance based on MCC and Accuracy.
- **Logistic Regression:** Preferred when higher Precision and AUC are important.
- **Decision Tree:** Preferred when a better balance between Precision and Recall, reflected by F1-score, is required.
- **Gaussian Naive Bayes:** Achieves the highest Recall but generates too many false positives.

Since the dataset is imbalanced, the high Accuracy values should be interpreted carefully. A model that predicts every session as a non-purchase would already achieve approximately 84.5% accuracy. Therefore, MCC, AUC, Precision, Recall, and F1-score provide additional insight into model performance.

---

# Additional Project Details

## 6. Data Preprocessing

The dataset contains numerical, categorical, and binary features. Different preprocessing techniques are applied according to the feature types.

### Numerical Features

The numerical features are standardized using `StandardScaler` to bring them onto a comparable scale.

### Categorical Features

Categorical features are transformed using `OneHotEncoder`. The encoder uses `handle_unknown='ignore'` to handle categories that may appear in test data but were not encountered during training.

The integer-coded features such as `OperatingSystems`, `Browser`, `Region`, and `TrafficType` are treated as categorical variables rather than continuous numerical values.

### Binary Features

The `Weekend` feature is retained using passthrough without additional transformation.

### Preprocessing Implementation

A `ColumnTransformer` is used to apply the appropriate transformations to each feature group.

The preprocessing steps are included within a scikit-learn `Pipeline` along with each classifier.

This ensures that:

- Preprocessing is fitted only on the training data.
- The same fitted transformations are applied to the test data.
- Data leakage from the test set is avoided.
- Each saved model contains its corresponding fitted preprocessing components.

The original 17 input features are transformed into 74 features after preprocessing.

**Preprocessing Artifacts:** The preprocessing components, including `StandardScaler`, `OneHotEncoder`, and `ColumnTransformer`, are encapsulated within each trained scikit-learn Pipeline. Saving each complete pipeline using Joblib preserves the fitted preprocessing transformations along with the trained classifier. Therefore, separate preprocessing artifact files are not required.

## 7. Train-Test Split

The dataset is divided into training and testing subsets using an 80:20 split.

The `train_test_split` function is configured with:

- `test_size=0.20`
- `random_state=42`
- `stratify=y`

Stratified sampling is used to maintain approximately the same target-class distribution in both subsets.

| Dataset      | Number of instances | Number of input features |
| ------------ | ------------------: | -----------------------: |
| Training set |               9,864 |                       17 |
| Testing set  |               2,466 |                       17 |
| **Total**    |          **12,330** |                   **17** |

The test dataset is exported as `test_data.csv` and is used by the Streamlit application to evaluate the trained models.

## 8. Streamlit Application

An interactive web application is developed using Streamlit to demonstrate the trained classification models and their performance.

**Live Application:** https://online-shopping-purchase-prediction-qgwqyqu9yq9cgrnc7sd6zf.streamlit.app/

### Application Features

The application provides the following functionality:

1. **CSV Upload:** Users can upload a test dataset in CSV format for evaluation.
2. **Dataset Preview:** Displays the uploaded dataset and its dimensions.
3. **Model Selection:** A dropdown allows users to select one of the five trained classification models.
4. **Evaluation Metrics:** Displays Accuracy, AUC, Precision, Recall, F1-score, and MCC for the selected model.
5. **Confusion Matrix:** Displays the confusion matrix to help understand correct and incorrect classifications.
6. **Classification Report:** Provides class-wise Precision, Recall, F1-score, and support.
7. **Model Comparison:** Displays the performance of all five models and provides metric-based visual comparisons.

The application loads the previously trained models and evaluates them on the uploaded test data. It does not retrain the models during application execution.

Each saved model includes its fitted preprocessing pipeline, so uploaded raw feature data can be passed directly to the selected model.

## 9. Repository Structure

The GitHub repository is organized as follows:

```text
online-shopping-purchase-prediction/
│
├── app.py
├── online_shoppers_model_training.ipynb
├── requirements.txt
├── README.md
├── test_data.csv
├── model_evaluation_results.csv
├── .python-version
├── .gitignore
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k-nearest_neighbors.joblib
    ├── gaussian_naive_bayes.joblib
    └── support_vector_machine.joblib
```

### Important Files

| File                                   | Purpose                                                                                                          |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `app.py`                               | Contains the Streamlit application source code.                                                                  |
| `online_shoppers_model_training.ipynb` | Contains the notebook used for dataset exploration, preprocessing, model training, evaluation, and model saving. |
| `requirements.txt`                     | Lists the Python dependencies required to run the application.                                                   |
| `README.md`                            | Provides project documentation, model results, and execution instructions.                                       |
| `test_data.csv`                        | Contains the held-out test dataset used by the Streamlit application.                                            |
| `model_evaluation_results.csv`         | Stores the evaluation metrics for all five models.                                                               |
| `.python-version`                      | Specifies Python 3.12 as the project Python version.                                                             |
| `model/`                               | Contains the five trained scikit-learn pipelines saved using Joblib.                                             |
| `.gitignore`                           | Specifies files and directories excluded from Git tracking.                                                      |

The original dataset can be downloaded from the UCI Machine Learning Repository using the link provided in the Dataset Description section.

## 10. Technologies Used

The project uses the following technologies and libraries:

| Technology / Library      | Purpose                                            |
| ------------------------- | -------------------------------------------------- |
| Python 3.12               | Programming language                               |
| NumPy                     | Numerical computations                             |
| Pandas                    | Data manipulation and analysis                     |
| Scikit-learn              | Data preprocessing, model training, and evaluation |
| Joblib                    | Saving and loading trained model pipelines         |
| Matplotlib                | Data visualization                                 |
| Seaborn                   | Statistical visualization                          |
| Streamlit                 | Interactive web application                        |
| Git and GitHub            | Version control and source code hosting            |
| Streamlit Community Cloud | Application deployment                             |

## 11. Running the Application Locally

Follow these steps to run the Streamlit application locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Rishabh-Singh-01/online-shopping-purchase-prediction
cd online-shopping-purchase-prediction
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available at the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

### Step 5: Upload Test Data

Upload a CSV file containing the required 17 input features and the `Revenue` target column. Select a trained model to view its evaluation metrics and classification results.

## 12. Deployment

The Streamlit application is deployed using **Streamlit Community Cloud**.

The deployment is connected to the GitHub repository, allowing the application to use the source code, dependencies, and saved model files directly from the repository.

The application uses Python 3.12 and the dependencies specified in `requirements.txt`.

Once deployed, users can access the application through the live Streamlit URL provided in Section 8.
