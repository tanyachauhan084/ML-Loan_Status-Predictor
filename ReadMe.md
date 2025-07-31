# 🏦 Loan Status Predictor

A Machine Learning project that predicts whether a loan will be approved based on applicant details. Built using Python, Pandas, Scikit-Learn, and Jupyter Notebook.

---


## 📊 Project Overview

This project aims to automate the loan approval process by predicting whether a loan will be granted or not, based on the applicant's personal and financial data. It is a binary classification problem where the target variable is whether the loan is approved (`Y`) or not (`N`).

---

## ❓ Problem Statement

Loan applications are typically processed manually, which is time-consuming and error-prone. Automating this process using machine learning can save time and provide accurate predictions. This model will help financial institutions make data-driven decisions regarding loan approvals.

---

## 📂 Dataset Description

The dataset contains 614 records with 13 features and 1 target column (`Loan_Status`).

| Feature              | Description                                    |
|----------------------|------------------------------------------------|
| Loan_ID              | Unique Loan ID                                 |
| Gender               | Male / Female                                  |
| Married              | Applicant married (Y/N)                        |
| Dependents           | Number of dependents                           |
| Education            | Graduate / Not Graduate                        |
| Self_Employed        | Self-employed (Y/N)                            |
| ApplicantIncome      | Applicant income                               |
| CoapplicantIncome    | Coapplicant income                             |
| LoanAmount           | Loan amount in thousands                       |
| Loan_Amount_Term     | Term of loan in months                         |
| Credit_History       | Credit history meets guidelines (1: Yes, 0: No)|
| Property_Area        | Urban / Semiurban / Rural                      |
| Loan_Status          | Target variable (Y: Approved, N: Rejected)     |

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

## 🔄 Project Pipeline

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Categorical Encoding
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Predictions
9. accuracy

---

## 📈 Exploratory Data Analysis

Key findings during EDA:

- Most applicants are male and married.
- Higher income applicants tend to get their loans approved.
- Credit history is a strong indicator of loan approval.
- Most approved loans are for applicants from semiurban areas.

---

## 🧹 Data Preprocessing

- **Missing Values** handled using dropna()
- **Categorical Encoding**: LabelEncoder and OneHotEncoder.
- **Train-Test Split**: 90% training, 10% testing.

---

## 🤖 Model Training

Several models were trained and compared:

- Logistic Regression ✅
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Final model selected: **Logistic Regression** for its simplicity and interpretability.


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# 📊 Model Evaluation

This model is evaluated using accuracy score

# ✅ Results

Accuracy: ~76% on test data
Most important feature: Credit History
Model generalizes well on unseen data


# 📌 Conclusion

The loan prediction model successfully identifies whether a loan application should be approved based on historical patterns. Logistic Regression provided a strong baseline and interpretable results. With more data and feature tuning, even better results can be achieved.


# ⚙️ How to Run

1. Clone this repository
2. Navigate to the project director
3. Install the required libraries
4. Run the Jupyter Notebook
  
# 🙋‍♀️ Author
Tanya Chauhan
🔗 LinkedIn : https://www.linkedin.com/in/tanya-chauhan-99a5aa355/
📧 tanyachauhan084@gmail.com


