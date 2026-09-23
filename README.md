# 📊 Telco Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn in the telecommunications industry, featuring exploratory data analysis, data preprocessing, model comparison, ensemble learning, and a Flask-based prediction application.

## 🚀 Project Overview

Customer churn is a major challenge for telecommunications companies. Identifying customers who are likely to leave can help businesses understand churn patterns and support targeted customer-retention strategies.

This project develops a machine learning pipeline that:

* Explores the Telco Customer Churn dataset
* Cleans and preprocesses customer data
* Compares multiple classification algorithms
* Builds a hard-voting ensemble model
* Evaluates models using precision, recall, accuracy, and F1-score
* Serializes the selected model
* Serves predictions through a Flask web application
* Retrieves the trained model from Amazon S3
* Provides a container configuration for deployment

The project is available on [GitHub](https://github.com/kikiopenifoluwa/Telco_churn_project?utm_source=chatgpt.com).

---

## 🎯 Problem Statement

Customer churn occurs when an existing customer stops using a company's services.

The objective of this project is to predict whether a telecom customer is likely to churn based on demographic, service, and account information.

The project uses customer characteristics such as:

* Gender
* Senior citizen status
* Partner and dependent status
* Tenure
* Phone services
* Internet services
* Online security and backup
* Device protection
* Technical support
* Streaming services
* Contract type
* Monthly charges
* Total charges

The target variable is:

```text
Churn
```

with two possible classes:

```text
Yes
No
```

---

# 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

The notebook contains **7,043 customer records and 21 columns**. The exploratory analysis identifies `Churn` as the target variable.

The dataset contains:

| Feature            | Description                              |
| ------------------ | ---------------------------------------- |
| `customerID`       | Unique customer identifier               |
| `gender`           | Customer gender                          |
| `SeniorCitizen`    | Whether the customer is a senior citizen |
| `Partner`          | Whether the customer has a partner       |
| `Dependents`       | Whether the customer has dependents      |
| `tenure`           | Number of months with the company        |
| `PhoneService`     | Whether phone service is active          |
| `MultipleLines`    | Multiple phone lines                     |
| `InternetService`  | Type of internet service                 |
| `OnlineSecurity`   | Online security service                  |
| `OnlineBackup`     | Online backup service                    |
| `DeviceProtection` | Device protection service                |
| `TechSupport`      | Technical support service                |
| `StreamingTV`      | Streaming TV service                     |
| `StreamingMovies`  | Streaming movie service                  |
| `Contract`         | Contract type                            |
| `PaperlessBilling` | Paperless billing status                 |
| `PaymentMethod`    | Payment method                           |
| `MonthlyCharges`   | Monthly customer charges                 |
| `TotalCharges`     | Total customer charges                   |
| `Churn`            | Target variable                          |

---

# 🔎 Exploratory Data Analysis

The notebook performs exploratory analysis before model development.

The dataset contains:

* **7,043 rows**
* **21 columns**
* **18 object columns**
* **2 integer columns**
* **1 float column**
* No missing values reported during the initial missing-value check.

The analysis also examines customer characteristics and churn-related patterns through visualizations and grouped analysis.

---

# 🧹 Data Preprocessing

The preprocessing logic is implemented in `preprocessing.py` and is also incorporated into the notebook's machine-learning pipeline.

### Total Charges

`TotalCharges` is converted from text to a numeric representation:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

The notebook performs this conversion before modeling.

### Feature Encoding

Categorical variables are converted into numerical representations suitable for machine learning.

Examples include:

```text
Yes / No
Male / Female
Internet service categories
Contract categories
```

The same preprocessing function is incorporated into the model pipeline using Scikit-learn's `FunctionTransformer`.

---

# 🤖 Machine Learning Models

The notebook evaluates six models:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors
5. Support Vector Classifier
6. Voting Classifier

The models are trained within a pipeline that applies the project's custom preprocessing before model fitting.

---

# 🧠 Ensemble Learning

The final application uses a **VotingClassifier**.

The ensemble uses hard voting and combines:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Support Vector Classifier

The notebook defines the ensemble with:

```python
VotingClassifier(
    voting="hard",
    estimators=[
        ("lr", LogisticRegression()),
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("rf", RandomForestClassifier()),
        ("knn", KNeighborsClassifier()),
        ("svc", SVC(probability=True, random_state=42))
    ]
)
```

The notebook subsequently identifies the VotingClassifier as the model used by the application.

---

# 📈 Model Performance

## Training Performance

The notebook's final model-comparison table reports the following **training-set metrics**:

| Model                     |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression       |     79.73% |     64.77% |     51.87% |     57.61% |
| Decision Tree             |     99.61% |     99.66% |     98.86% |     99.26% |
| Random Forest             |     99.59% |     99.00% |     99.47% |     99.23% |
| K-Nearest Neighbors       |     82.93% |     73.92% |     55.15% |     63.17% |
| Support Vector Classifier |     73.45% |      0.00% |      0.00% |      0.00% |
| **Voting Classifier**     | **91.52%** | **99.61%** | **68.32%** | **81.05%** |

> **Important:** These are training metrics. The extremely high Decision Tree and Random Forest training scores should therefore not be interpreted as their expected performance on unseen customers.

---

## 🧪 Test Performance

The notebook also generates classification reports on a **1,409-row test set**.

For the selected VotingClassifier, the test results were:

| Metric    | No Churn | Churn |
| --------- | -------: | ----: |
| Precision |      82% |   69% |
| Recall    |      93% |   42% |
| F1 Score  |      87% |   52% |
| Support   |    1,036 |   373 |

Overall test accuracy:

**79%**

```text
              precision    recall  f1-score   support

No               0.82      0.93      0.87      1036
Yes              0.69      0.42      0.52       373

accuracy                              0.79      1409
macro avg         0.75      0.67      0.69      1409
weighted avg      0.78      0.79      0.78      1409
```

The test results show an important distinction between overall accuracy and the model's ability to identify the positive churn class: the model achieved **69% precision and 42% recall for customers who actually churned**.

---

# 🏆 Selected Model

The **VotingClassifier** is the model used by the deployed application.

The notebook's model-evaluation section explains that the VotingClassifier was selected because it combines multiple classifiers and emphasizes precision as the primary evaluation metric for the project's stated objective.

The selected model's key results are:

```text
Training Accuracy   : 91.52%
Training Precision  : 99.61%
Training Recall     : 68.32%
Training F1 Score   : 81.05%

Test Accuracy       : 79.00%
Test Churn Precision: 69.00%
Test Churn Recall   : 42.00%
Test Churn F1       : 52.00%
```

These figures are reported directly from the executed notebook outputs.

---

# ⚠️ Model Performance Considerations

There is a noticeable gap between the training and test results.

For example, the VotingClassifier has:

```text
Training F1: 81.05%
Test Churn F1: 52%
```

and:

```text
Training Recall: 68.32%
Test Churn Recall: 42%
```

This indicates that performance on unseen data is substantially lower than the training metrics.

The same issue is especially apparent for the Decision Tree and Random Forest models, whose training accuracy and F1 scores are approximately 99%.

Therefore, the test-set results are particularly important when interpreting the model's ability to identify future churn.

---

# 🌐 Flask Web Application

The prediction application is implemented using **Flask**.

The main application is:

```text
main.py
```

The application provides a web form through:

```text
templates/main.html
```

Users can provide customer information, which is transformed into the format expected by the trained model.

The application then returns a churn prediction.

---

# ☁️ AWS S3 Model Storage

The Flask application retrieves the trained model from **Amazon S3** using `boto3`.

The model artifact used by the application is:

```text
Telco_model_colab.pkl
```

The application is configured to retrieve this model from an S3 bucket.

For production deployment, AWS credentials should be supplied securely through:

* IAM roles
* Environment variables
* AWS credential profiles
* A managed secrets solution

**AWS credentials should never be committed to GitHub.**

---

# 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │   Customer Input     │
                    │     HTML Form        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Flask App      │
                    │      main.py        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Data Preprocessing │
                    │ preprocessing.py    │
                    └──────────┬──────────┘
                               │
                               ▼
             ┌───────────────────────────────────┐
             │         Voting Classifier         │
             │                                   │
             │ Logistic Regression               │
             │ Decision Tree                     │
             │ Random Forest                     │
             │ KNN                               │
             │ SVM                               │
             └────────────────┬──────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Churn Prediction  │
                    └─────────────────────┘

                         AWS S3
                           │
                           ▼
                  Telco_model_colab.pkl
```

---

# 📁 Project Structure

```text
Telco_churn_project/
│
├── templates/
│   └── main.html
│
├── .gitignore
├── Containerfile
├── main.py
├── preprocessing.py
├── requirements.txt
└── telco_churn.py
```

The repository also contains the project notebook `Telco_Churn.ipynb`, which documents the exploratory analysis, preprocessing, model training, evaluation, and model-selection process.

### File Descriptions

| File                  | Purpose                                                 |
| --------------------- | ------------------------------------------------------- |
| `Telco_Churn.ipynb`   | Exploratory analysis, model development and evaluation  |
| `telco_churn.py`      | Python implementation of the model-development workflow |
| `preprocessing.py`    | Data preprocessing and feature transformation           |
| `main.py`             | Flask prediction application                            |
| `templates/main.html` | Web interface                                           |
| `requirements.txt`    | Python dependencies                                     |
| `Containerfile`       | Container configuration                                 |
| `.gitignore`          | Git exclusions                                          |

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Support Vector Machine
* Voting Classifier

### Web Development

* Flask
* HTML

### Cloud

* Amazon S3
* Boto3

### Deployment

* Gunicorn
* Containerfile

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/kikiopenifoluwa/Telco_churn_project.git
cd Telco_churn_project
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Flask application:

```bash
python main.py
```

The application is configured to run on:

```text
http://localhost:5000
```

---

# 🧪 Running the Model Development Workflow

The model-development workflow is contained in:

```text
telco_churn.py
```

The notebook version is:

```text
Telco_Churn.ipynb
```

The notebook loads:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

and performs:

```text
Data Loading
     ↓
Exploratory Data Analysis
     ↓
Data Cleaning
     ↓
Feature Preprocessing
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Voting Classifier
     ↓
Model Serialization
```

---

# 🐳 Containerization

The repository includes a `Containerfile`.

Build the container with Podman:

```bash
podman build -t telco-churn .
```

Run it:

```bash
podman run -p 5000:5000 telco-churn
```

Then access:

```text
http://localhost:5000
```

---

# 💼 Business Use Case

A churn prediction system can support telecom businesses by helping them:

* Identify customers at risk of leaving
* Analyze customer churn patterns
* Support customer-retention campaigns
* Understand customer characteristics associated with churn
* Prioritize customers for further investigation

The prediction should be considered a **decision-support signal**, rather than a guarantee that a particular customer will churn.

---

# 🔮 Future Improvements

Potential improvements to the project include:

* Hyperparameter tuning for the individual classifiers
* Cross-validation
* Better handling of class imbalance
* Threshold optimization for churn detection
* Probability-based churn predictions
* SHAP-based model explainability
* Automated model evaluation
* Model versioning
* Experiment tracking
* Input validation in the Flask application
* Automated testing
* CI/CD with GitHub Actions
* Model monitoring
* Data-drift monitoring
* REST API endpoints
* Improved cloud deployment

---

# 👤 Author

**Kiki Openifoluwa**

GitHub:
[@kikiopenifoluwa](https://github.com/kikiopenifoluwa?utm_source=chatgpt.com)

Project:
[Telco Churn Project](https://github.com/kikiopenifoluwa/Telco_churn_project?utm_source=chatgpt.com)

---

# 📄 License

No license file is currently included in the repository.

If you intend to distribute the project as open source, consider adding an appropriate license.

---

## ⭐ Project Summary

This project demonstrates an end-to-end machine learning workflow:

```text
Business Problem
       ↓
Data Exploration
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Model Development
       ↓
Model Comparison
       ↓
Ensemble Learning
       ↓
Model Evaluation
       ↓
Model Serialization
       ↓
AWS S3
       ↓
Flask Application
       ↓
Containerized Deployment
```

The executed notebook shows that the final **VotingClassifier achieved 91.52% training accuracy and 79% test accuracy**, with **69% precision, 42% recall, and 52% F1-score for the churn class on the test set**.
