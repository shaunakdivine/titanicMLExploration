# titanicMLExploration
Titanic Survival Prediction with Gradient Boosting: This project applies Gradient Boosting to predict passenger survival on the Titanic, featuring data preprocessing, model training with caret, and performance evaluation through ROC, precision-recall curves, and more. Explore the repository for scripts, results, and insights.

# Titanic Survival Prediction with Gradient Boosting
Project Overview
This project applies a Gradient Boosting Machine (GBM) model to predict passenger survival on the Titanic. The primary focus is on data preprocessing, model training with caret, and performance evaluation through various metrics and visualization techniques.

# Key Features
Data Cleaning and Preparation: The Titanic dataset is cleaned and preprocessed to handle missing values, encode categorical variables, and create new features like Title and Deck_Level.
Model Training: A GBM model is trained using the caret package in R, with cross-validation for model tuning and selection.
Performance Evaluation: The model's performance is evaluated using metrics like Accuracy, Kappa, AUC-ROC, TPR (True Positive Rate), FPR (False Positive Rate), and Log Loss. Additionally, ROC curves, precision-recall curves, and lift charts are plotted for visual analysis.
Feature Importance: The importance of various features is visualized to understand the model's decision-making process.
Repository Contents
titanic_cleaned.csv: Cleaned Titanic dataset used for training and testing the model.
titanic_boosting.R: R script containing the entire data pipeline, including data preprocessing, model training, and performance evaluation.
plots/: Directory containing generated plots, such as ROC curves, precision-recall curves, and feature importance charts.
README.md: This file, providing an overview of the project.

# Model Performance Summary
Accuracy: 0.84
Kappa: 0.66
AUC-ROC: 0.87
TPR (Sensitivity): 0.78
FPR (1-Specificity): 0.12
Log Loss: 0.68
These results indicate that the model performs well in distinguishing between survivors and non-survivors, with a relatively low false positive rate.

# Key Conclusions
The GBM model successfully predicts Titanic passenger survival with high accuracy and a strong AUC-ROC.
The low FPR compared to the TPR suggests that the model is effective at minimizing false alarms, which is crucial in classification tasks.
Feature importance analysis highlights which variables have the most influence on survival predictions, providing insights into the factors that contributed to survival during the Titanic disaster.
