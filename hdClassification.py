
import pandas as pd
import matplotlib.pyplot as plt  #ploting
import seaborn as sns  # For enhanced visualizations
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

#Question 1
# Load the dataset using pandas
df = pd.read_csv(r'C:\Users\20021\OneDrive\Desktop\VS Code\heart_disease_dataset.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Define the list of variables
variables = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']

# Loop over each variable to create a histogram with a Kernel Density Estimate (KDE)
for var in variables:
    plt.figure()  #new figure for each plot
    sns.histplot(df[var], kde=True)  # 'kde=True' for smooth density curve
    plt.title(f'Histogram of {var}')  # f-string for dynamic string formatting
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

# Calculate and print the skewness for each variable
print("Skewness of each variable:")
for var in variables:
    skew_value = df[var].skew()  # pandas provides a .skew() method
    print(f"{var}: {skew_value:.2f}")

n_total = len(df)
print("Total samples:", n_total)

#80% for training 20% for testing
n_train = 734
train_set = df.iloc[:n_train]
test_set = df.iloc[n_train:]
print("Training size:", len(train_set))
print("Test size:", len(test_set))
print("Proportions of heart disease in the training:")
train_proportions = train_set["HeartDisease"].value_counts(normalize=True)
print(train_proportions)
print("Proportions of heart disease in the test set:")
test_proportions = test_set["HeartDisease"].value_counts(normalize=True)
print(test_proportions)

#KNN
features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
target = 'HeartDisease'
#extract features and target for training and testing
X_train = train_set[features]
y_train = train_set[target]
X_test = test_set[features]
y_test = test_set[target]
#scale the features for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#KNN 5
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred = knn_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("KNN Model Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

#KNN 1 to 10
k_values = range(1, 11)
accuracies = []
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)
    y_pred = knn_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Plotting K
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o')
plt.title('K vs. Accuracy')
plt.xlabel('k')
plt.ylabel('Test Accuracy')
plt.xticks(k_values)
plt.show()
best_k = k_values[accuracies.index(max(accuracies))]
print("Best K:", best_k)
print("Highest test Accuracy:", max(accuracies))

# SVM
svm_model = SVC(probability=True, random_state=461)
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("SVM Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

y_train_prob = svm_model.predict_proba(X_train_scaled)[:, 1]
y_test_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
#ROC curves
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
#AUCs
auc_train = roc_auc_score(y_train, y_train_prob)
auc_test = roc_auc_score(y_test, y_test_prob)
# Plot both curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f'Training AUC = {auc_train:.3f}')
plt.plot(fpr_test, tpr_test, label=f'Testing AUC = {auc_test:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve for SVM on Training and Testing Sets')
plt.legend(loc='lower right')
plt.show()