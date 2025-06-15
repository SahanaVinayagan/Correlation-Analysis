import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = 'diabetes_dataset.csv'
diabetes_data = pd.read_csv(file_path)

# Preprocessing
categorical_columns = diabetes_data.select_dtypes(include='object').columns
encoder = LabelEncoder()

for col in categorical_columns:
    diabetes_data[col] = encoder.fit_transform(diabetes_data[col])

# Separate features and target variable
X = diabetes_data.drop('Target', axis=1)
y = diabetes_data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute correlation matrix
correlation_matrix = diabetes_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix of Diabetes Dataset")
plt.show()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
random_forest.fit(X_train, y_train)

# Predict and Evaluate
y_pred = random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation results
print("Random Forest Classifier Results:")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

# Feature Importance
feature_importances = random_forest.feature_importances_
feature_names = X.columns
# Plot feature importances
sorted_indices = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(16, 10))
sns.barplot(y=feature_names[sorted_indices][:10],
            x=feature_importances[sorted_indices][:10],
            palette="viridis",
            hue=feature_names[sorted_indices][:10],
            legend=False)
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.show()