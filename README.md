# Credit Card Fraud Detection
A machine learning project for detecting fraudulent credit card transactions on a highly imbalanced dataset.

# Overview
This project builds and compares classification models to identify fraudulent transactions among ~200,000 credit card transactions, where fraud cases make up less than 0.2% of the data. The main challenge addressed is severe class imbalance.

# Dataset
- ~200,000 transaction records
- Target column: `isFraud` (binary — fraud / not fraud)
- Highly imbalanced: 282 fraud cases out of ~200,000 transactions

# Approach

1. **Preprocessing**
   - Feature scaling with `StandardScaler`
   - Categorical encoding with `LabelEncoder`
   - Train/test split

2. **Handling Class Imbalance**
   - Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training set

3. **Modeling**
   - Trained and compared two classifiers:
     - **Random Forest** (`n_estimators=100`, `class_weight='balanced_subsample'`)
     - **K-Nearest Neighbors** (`n_neighbors=5`)

4. **Evaluation**
   - Accuracy, precision, recall, F1-score (via classification report)
   - ROC-AUC score
   - Confusion matrix visualization

# Results

| Model | Accuracy | ROC-AUC | Recall (fraud class) |
|---|---|---|---|
| Random Forest | 99.87% | 0.999 | 0.98 |
| KNN | 98.90% | 0.907 | 0.79 |

Random Forest significantly outperformed KNN, particularly in ROC-AUC and precision on the minority (fraud) class, making it the better-suited model for this imbalanced classification task.

## Tech Stack
- Python
- pandas, NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

# Notes
This was built as a course project, extending exploratory work started during an ML training program.
