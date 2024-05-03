import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV file
file_path = r"C:\Users\ASUS\Desktop\CVD_cleaned (1).csv"  # Update with the path to your CSV file
data = pd.read_csv(file_path)

# Data preprocessing
# Encoding categorical variables
le = LabelEncoder()
data['General_Health'] = le.fit_transform(data['General_Health'])
data['Checkup'] = le.fit_transform(data['Checkup'])
data['Exercise'] = le.fit_transform(data['Exercise'])
data['Skin_Cancer'] = le.fit_transform(data['Skin_Cancer'])
data['Other_Cancer'] = le.fit_transform(data['Other_Cancer'])
data['Depression'] = le.fit_transform(data['Depression'])
data['Diabetes'] = le.fit_transform(data['Diabetes'])
data['Arthritis'] = le.fit_transform(data['Arthritis'])
data['Sex'] = le.fit_transform(data['Sex'])
data['Age_Category'] = le.fit_transform(data['Age_Category'])
data['Smoking_History'] = le.fit_transform(data['Smoking_History'])

# Feature selection
X = data[['General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 'Other_Cancer', 'Depression',
          'Diabetes', 'Arthritis', 'Sex', 'Age_Category']]
y = data['Heart_Disease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize classification report
plt.figure(figsize=(8, 6))
report = classification_report(y_test, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="YlGnBu")
plt.title('Classification Report')
plt.show()
