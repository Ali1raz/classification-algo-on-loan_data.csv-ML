import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

matplotlib.use('Agg')

# Step i: load the dataset
file_path = "loan_data.csv"
data = pd.read_csv(file_path)

# Step ii: Preprocess the data
# Encoding categorical variables
categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separating features and target
X = data.drop(['loan_status'], axis=1)
y = data['loan_status']

# Encode target variable
y = LabelEncoder().fit_transform(y)

# Step iii: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step iv: train a Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step v: make predictions
y_pred = clf.predict(X_test)

# Step vi: Evaluate the model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["No", "Yes"], zero_division=1)

# display metrix
print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(report)

# visualize confusion matrix and save as image
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# save the plot as png
plt.savefig("output_plots/decision_tree_cm.png")
print("Confusion matrix plot saved as 'output_plots/decision_tree_cm.png'.")
