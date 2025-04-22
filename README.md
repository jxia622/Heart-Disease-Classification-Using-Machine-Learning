# Heart-Disease-Classification-Using-Machine-Learning
Project Overview
This project applies machine learning techniques to classify heart disease based on clinical features. Two models — Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) — were developed and evaluated for their ability to predict the presence of heart disease from patient data.

The dataset contains five variables: age, resting blood pressure, cholesterol, maximum heart rate, and heart disease diagnosis (binary outcome).

Key steps include:

Exploratory Data Analysis (EDA) of patient features

Model training using an 80/20 train-test split

Model evaluation using cross-validation, confusion matrices, ROC curves, and performance metrics (accuracy, precision, recall, AUC)

Methods
EDA:

Age: Approximately normal distribution with slight left skewness.

Resting Blood Pressure: Heavily left-skewed.

Cholesterol: Right-skewed with an unusual peak at zero.

Max Heart Rate: Approximately normal distribution with a dip in the middle.

Data Splitting:

80% (n = 734) for training, 20% (n = 184) for testing.

Training heart disease proportion: 57.5% diseased, 42.5% non-diseased.

Testing heart disease proportion: 53.3% diseased, 46.7% non-diseased.

Model 1: K-Nearest Neighbors (KNN)

Iterated K from 1 to 10.

Best K = 5, achieving highest test set accuracy.

Confusion Matrix for K = 5:

True Negatives (TN): 72

False Positives (FP): 26

False Negatives (FN): 35

True Positives (TP): 51

Test Accuracy: 66.8%

KNN showed better recall but lower precision compared to SVM.

Model 2: Support Vector Machine (SVM)

Trained using scaled features with random state = 461.

Confusion Matrix:

True Negatives (TN): 82

False Positives (FP): 16

False Negatives (FN): 40

True Positives (TP): 46

Test Accuracy: 69.6%

ROC AUC:

Training AUC: 0.775

Testing AUC: 0.742

Results Summary

Model	Test Accuracy	AUC (Test)	Key Observations
KNN (K=5)	66.8%	[Not calculated]	Higher recall but lower precision
SVM	69.6%	0.742	Better precision, fewer false positives
Based on overall metrics, the SVM model showed slightly better performance, offering higher precision and fewer false positives compared to KNN.

Notes
The heart disease dataset used was provided by CWRU for educational purposes.

If further explanation, full reports, or demonstrations are needed, please feel free to contact me. cxxia169@gmail.com
