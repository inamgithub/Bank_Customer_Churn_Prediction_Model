# Bank Customer Churn Prediction Model

This repository contains a Jupyter Notebook that demonstrates the process of building a customer churn prediction model for a bank using Support Vector Machine (SVM) classification. The model aims to predict whether a customer is likely to churn or not, based on various features provided in the dataset.

## Learning Objectives
The code in the Jupyter Notebook covers the following learning objectives:

### Data Encoding: 
The categorical variables in the dataset, such as "Geography" and "Gender," are encoded into numeric values to be used in the SVM model.

### Feature Scaling: 
The numerical features in the dataset are standardized using the StandardScaler to ensure that all features have the same scale.

### Handling Imbalance Data: 
The dataset is checked for class imbalance, where one class (e.g., churned customers) significantly outnumbers the other. The code demonstrates two techniques to address this issue: Random Under Sampling and Random Over Sampling.

### Support Vector Machine Classifier:
An SVM classifier is applied to the data for predicting customer churn. The SVM algorithm is chosen for its ability to handle both linear and non-linear decision boundaries.

### Grid Search for Hyperparameter Tuning:
Grid search is performed to find the optimal hyperparameters for the SVM classifier. This process helps to fine-tune the model and improve its performance.

## Dataset
The dataset used for this project is named "Bank Churn Modelling.csv," and it contains various features related to bank customers, including their credit score, age, tenure, balance, estimated salary, geography, gender, and more.

## Getting Started
To run the code and reproduce the churn prediction model:

Clone this repository to your local machine or download the "Bank Customer Churn Model.ipynb" file.

Install the required libraries by running the following command:

### Copy code
### pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
### Open the Jupyter Notebook "Bank Customer Churn Model.ipynb" using Jupyter Notebook or Jupyter Lab.

Execute each cell in the notebook to step through the data encoding, handling imbalance data, feature scaling, SVM model training, hyperparameter tuning, and model evaluation.

## Model Evaluation
The performance of the churn prediction model is evaluated for three different scenarios:

### Original Data: 
The model is trained and evaluated on the original imbalanced dataset.

### Random Under Sampling: 
The model is trained on the randomly under-sampled dataset to handle class imbalance.

### Random Over Sampling: 
The model is trained on the randomly over-sampled dataset to handle class imbalance.

The evaluation metrics include confusion matrices and classification reports, providing insights into the model's accuracy, precision, recall, and F1 score for each scenario.

## Conclusion
The churn prediction model built using SVM classification can be valuable for banks to identify customers at a higher risk of churn and implement targeted retention strategies. By reducing customer churn, banks can improve customer satisfaction, loyalty, and overall business profitability.
