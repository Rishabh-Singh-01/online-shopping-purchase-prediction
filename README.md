# Online Shopping Purchase Intention Prediction

## 1. Project Overview

This project focuses on predicting whether an online visitor will generate revenue (make a purchase) during a browsing session.

The objective is to build and evaluate multiple machine learning classification models using the Online Shoppers Purchasing Intention dataset. Five classification algorithms are implemented and compared using six evaluation metrics.

An interactive Streamlit application is also developed to allow users to upload test data, select a trained model, and view its evaluation results.

**GitHub Repository:** https://github.com/Rishabh-Singh-01/online-shopping-purchase-prediction

**Live Streamlit Application:** https://online-shopping-purchase-prediction-qgwqyqu9yq9cgrnc7sd6zf.streamlit.app/

## 2. Problem Statement

Online shopping platforms collect information about visitors' browsing activities, such as the number of pages visited, time spent on different pages, visitor type, and other session-related characteristics.

The goal of this project is to use these features to predict whether a visitor will make a purchase.

This is formulated as a binary classification problem:

- `True`: The visitor makes a purchase.
- `False`: The visitor does not make a purchase.

The project compares different classification algorithms to understand their performance and suitability for this prediction task.

## 3. Dataset Description

The project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

**Dataset source:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

| Attribute       | Description           |
| --------------- | --------------------- |
| Total instances | 12,330                |
| Input features  | 17                    |
| Target variable | Revenue               |
| Problem type    | Binary classification |
| Missing values  | None                  |

### Features

The dataset contains numerical, categorical, and binary features.

- **Numerical features:** Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay.
- **Categorical features:** Month, OperatingSystems, Browser, Region, TrafficType, VisitorType.
- **Binary feature:** Weekend.
- **Target variable:** Revenue.

### Target Distribution

The target variable is imbalanced:

| Class               | Instances | Percentage |
| ------------------- | --------: | ---------: |
| False (No purchase) |    10,422 |     84.53% |
| True (Purchase)     |     1,908 |     15.47% |

This class imbalance is considered when evaluating model performance, since accuracy alone may not provide a complete picture of classification effectiveness.

## 4. Data Preprocessing

The following preprocessing steps were applied:

1. Separated the input features from the target variable (`Revenue`).
2. Identified numerical, categorical, and binary features.
3. Applied StandardScaler to numerical features.
4. Applied OneHotEncoder to categorical features.
5. Passed binary features through without transformation.
6. Combined preprocessing and classification using scikit-learn pipelines.

The preprocessing transformation expanded the original 17 input features into 74 processed features.

### Train-Test Split

The dataset was divided into training and testing sets using an 80:20 split.

- Training samples: 9,864
- Testing samples: 2,466
- Random state: 42
- Stratification: Enabled

Stratification was used to preserve the target class distribution in both sets.

## 5. Classification Models

Five classification algorithms were implemented:

| Model                  | Configuration                             |
| ---------------------- | ----------------------------------------- |
| Logistic Regression    | Maximum iterations: 1000                  |
| Decision Tree          | Random state: 42                          |
| K-Nearest Neighbors    | K = 5                                     |
| Gaussian Naive Bayes   | GaussianNB                                |
| Support Vector Machine | RBF kernel, probability estimates enabled |

Each model was trained using the same training dataset and preprocessing pipeline.

The trained pipelines were saved using Joblib and reused by the Streamlit application. The application does not retrain the models.

## 6. Model Evaluation

The models were evaluated using six metrics:

- **Accuracy:** Overall proportion of correctly classified observations.
- **AUC:** Measures the model's ability to distinguish between the two classes across classification thresholds.
- **Precision:** Proportion of predicted purchases that were actual purchases.
- **Recall:** Proportion of actual purchases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Measures the overall quality of binary classifications while accounting for all four confusion matrix outcomes.

### Performance Comparison

The following results were obtained on the held-out test dataset.

| Model                  |   Accuracy |        AUC |  Precision |     Recall |         F1 |        MCC |
| ---------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression    |     0.8812 | **0.8830** | **0.7405** |     0.3586 |     0.4832 |     0.4610 |
| Decision Tree          |     0.8646 |     0.7435 |     0.5622 | **0.5681** | **0.5651** |     0.4849 |
| K-Nearest Neighbors    |     0.8715 |     0.7572 |     0.6419 |     0.3848 |     0.4812 |     0.4307 |
| Gaussian Naive Bayes   |     0.2733 |     0.7330 |     0.1727 |     0.9738 |     0.2934 |     0.1292 |
| Support Vector Machine | **0.8873** |     0.8817 |     0.7063 |     0.4660 |     0.5615 | **0.5142** |

### Observations

**1. Logistic Regression**

- Achieved the highest AUC of 0.8830 and highest precision of 0.7405.
- Its relatively high precision indicates that it produces fewer false purchase predictions.
- However, its recall of 0.3586 indicates that it misses a considerable proportion of actual purchases.
- It can be useful when reducing false positive purchase predictions is important.

**2. Decision Tree**

- Achieved the highest recall of 0.5681 and highest F1-score of 0.5651 among the five models.
- It identifies more actual purchases than the other models, except Gaussian Naive Bayes.
- Its accuracy and AUC are lower than those of Logistic Regression and SVM.
- It provides a useful alternative when identifying more potential purchasers is a priority.

**3. K-Nearest Neighbors**

- Achieved an accuracy of 0.8715 and precision of 0.6419.
- Its recall of 0.3848 and F1-score of 0.4812 indicate comparatively weaker performance in identifying actual purchases.
- It did not achieve the highest value in any of the six evaluation metrics.

**4. Gaussian Naive Bayes**

- Achieved the highest recall of 0.9738, but its precision was only 0.1727.
- Its very low accuracy of 0.2733 and MCC of 0.1292 indicate poor overall classification performance.
- It predicts a large number of non-purchasing visitors as purchasers, resulting in many false positives.
- Despite its high recall, its overall performance is not satisfactory.

**5. Support Vector Machine**

- Achieved the highest accuracy of 0.8873 and highest MCC of 0.5142.
- Its F1-score of 0.5615 is also competitive with the Decision Tree model.
- It provides a relatively balanced combination of precision and recall.
- Based on the overall classification performance, it is the strongest model among the five evaluated.

### Overall Conclusion

The Support Vector Machine achieved the best overall performance based on accuracy and Matthews Correlation Coefficient.

However, the choice of the best model depends on the intended application:

- **SVM:** Best overall balance based on the evaluation results.
- **Decision Tree:** More suitable when recall and F1-score are the priority.
- **Logistic Regression:** More suitable when precision and reducing false positives are important.

Since the dataset is imbalanced, accuracy alone is not sufficient to determine the best model. MCC, precision, recall, F1-score, and AUC provide additional insight into classification performance.

## 7. Streamlit Application

An interactive web application was developed using Streamlit to evaluate the trained classification models.

### Application Features

- Upload a test dataset in CSV format.
- Select one of the five trained classification models.
- Display six evaluation metrics: Accuracy, AUC, Precision, Recall, F1-score, and MCC.
- Display the confusion matrix.
- Display the classification report.
- Compare the performance of all five models on the uploaded dataset.

The application uses the saved model pipelines and their associated preprocessing steps. It does not perform model training during application execution.

The uploaded CSV must contain the 17 input features and the `Revenue` target column.

## 8. Repository Structure

```text
online-shopping-purchase-prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_evaluation_results.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k_nearest_neighbors.joblib
    ├── gaussian_naive_bayes.joblib
    └── support_vector_machine.joblib
```

## 9. Technologies Used

- Python 3.12
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 10. Running the Application Locally

### Prerequisites

Python 3.12 is recommended.

### Installation

Clone the repository:

```bash
git clone https://github.com/Rishabh-Singh-01/online-shopping-purchase-prediction
cd online-shopping-purchase-prediction
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## 11. Deployment

The application is deployed using Streamlit Community Cloud.

**Live Application:** https://online-shopping-purchase-prediction-qgwqyqu9yq9cgrnc7sd6zf.streamlit.app/

**Source Code:** https://github.com/Rishabh-Singh-01/online-shopping-purchase-prediction
