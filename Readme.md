''' Correlation Analysis of Risk Factors of Diabetes using Machine Learning

This project analyzes various health and lifestyle factors to determine their correlation with diabetes using a machine learning approach. By applying data preprocessing, correlation analysis, and a Random Forest Classifier, the project identifies the most significant risk factors and predicts diabetes outcomes with measurable accuracy.

Dataset
The dataset contains patient health records with attributes such as age, glucose levels, blood pressure, cholesterol, family history, lifestyle indicators, and more. The target variable Target indicates the presence (1) or absence (0) of diabetes.
Dataset Used: diabetes_dataset.csv
Source: Downloaded from Kaggle

Objectives
•Explore correlations between input features and diabetes status.
•Build a machine learning model to predict diabetes.
•Identify the most influential features contributing to diabetes.

Tools & Technologies
•Python
•Pandas 
•NumPy
•Matplotlib 
•Seaborn
•Scikit-learn

Project Workflow

1. Data Preprocessing
   •Categorical columns are encoded using `LabelEncoder`.
   •Numerical features are standardized using `StandardScaler`.

2. Correlation Analysis
   •A correlation matrix is plotted to analyze the relationships between variables.
   •Heatmap visualization highlights positive and negative correlations.

3. Model Building
   •Data is split into training and testing sets.
   •A Random Forest Classifier is trained for classification.

4. Model Evaluation
   •Metrics used: Accuracy Score, Classification Report, Confusion Matrix.
   •Prediction results are interpreted to assess model performance.

5. Feature Importance
   •Visualizes the top 10 most important features contributing to diabetes prediction.

Project Structure
Correlation Analysis/
│
├── diabetes_dataset.csv 
├── correlation_analysis.py 
├── README.md
├── requirements.txt 
├── Images/
    ├── correlation.png 
    └── top10.png 
 

How to Run the Project

This project can be executed using Jupyter Notebook or any Python IDE:

1. Open the correlation_analysis.py file (this contains the main Python program).

2. Copy the code and paste it into a Jupyter Notebook, or run the file directly using a Python IDE.

3. Make sure the diabetes_dataset.csv file is in the same directory, or upload it in your Jupyter environment.

4. Execute the code. It will:

     •Generate a correlation heatmap

     •Display the accuracy of the model

     •Plots a graph showing the top 10 important features

'''
