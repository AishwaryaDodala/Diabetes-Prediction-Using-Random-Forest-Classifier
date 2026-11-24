# Diabetes-Prediction-Using-Random-Forest-Classifier
# 📌 Project Definition
Diabetes is a chronic health condition that requires early identification and proper management. Identifying individuals who may be at risk can support further medical evaluation and timely intervention.

The objective of this project is to build a binary classification model that predicts the presence or absence of diabetes using patient-related attributes such as glucose level, blood pressure, BMI, insulin, age, and diabetes pedigree function.

The target variable is:

0: Non-diabetic

1: Diabetic

The project explores different Machine Learning algorithms to identify a suitable model for diabetes prediction.
### Dataset Overview
Diabetes Dataset 

The dataset contains 768 records and 9 columns, including 8 independent variables and 1 target variable.
# 📁 Methodology
The project follows a structured Machine Learning workflow.

### Step 1: Data Loading and Exploration

Imported the dataset using Pandas.

Examined dataset dimensions, column names, data types and sample records.

Checked for missing values and explored the distribution of variables.

### Step 2: Data Cleaning and Outlier Detection

Checked for missing values using isna().sum().

Used Seaborn boxplots to visualize potential outliers.

Applied the Interquartile Range (IQR) method to identify and cap extreme values.

Used lower and upper IQR boundaries to replace values outside the calculated range.

### Step 3: Feature and Target Separation

Separated independent variables (X) from the dependent variable (y).

Used the Outcome column as the target variable.

Retained eight input features for model training.

### Step 4: Handling Class Imbalance
Examined the distribution of the target variable.

Identified an imbalance between diabetic and non-diabetic records.

Applied RandomOverSampler to balance the dataset.

Balanced dataset distribution:

Non-diabetic: 500 records

Diabetic: 500 records

### Step 5: Train-Test Splitting

Divided the resampled dataset into training and testing sets.

Training data: 80%

Testing data: 20%

Used random_state=2 for the initial split.

### Step 6: Model Building

Implemented and compared five Machine Learning classification algorithms:

Decision Tree Classifier

Random Forest Classifier

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Logistic Regression

Random Forest achieved the highest accuracy among the evaluated models.
### Step 7: Cross-Validation

Applied 20-fold cross-validation to evaluate model performance across multiple data splits.

Compared the average cross-validation scores of the five algorithms.

Random Forest achieved the highest average score.

### Step 8: Hyperparameter Tuning

Applied two hyperparameter optimization techniques:

### GridSearchCV

Tested different combinations of hyperparameters.

Parameters included:

Number of estimators

Maximum tree depth

Splitting criterion

Best parameters:

n_estimators: 100

max_depth: 12

criterion: entropy

Best cross-validation score: 87.1%
### RandomizedSearchCV

Evaluated 15 randomly selected hyperparameter combinations.

Used 10-fold cross-validation.

Best parameters:

n_estimators: 150

max_depth: 12

criterion: log_loss

Best cross-validation score: 86.9%

# 🛠️ Technologies Used

Programming Language

Python

### Libraries and Frameworks

Pandas – Data manipulation and analysis

NumPy – Numerical computations

Matplotlib – Data visualization

Seaborn – Statistical visualization

Scikit-learn – Machine Learning model development and evaluation

Imbalanced-learn – Handling class imbalance using RandomOverSampler

### Machine Learning Algorithms

Random Forest

Decision Tree

K-Nearest Neighbors

Support Vector Machine

Logistic Regression

### Model Optimization

GridSearchCV

RandomizedSearchCV

Cross-validation

### Development Environment

Jupyter Notebook

# Evaluation Metrics and Model Performance

The models were evaluated using classification accuracy and cross-validation scores.

### Model Comparison

The following classification algorithms were evaluated:

```text
Algorithm                  Accuracy
-----------------------------------
Decision Tree              86%
Random Forest              93%
K-Nearest Neighbors        74%
Support Vector Machine     79%
Logistic Regression        78%
```

### Cross-Validation Results

```text
Algorithm                  Average CV Score
-------------------------------------------
Decision Tree              83.5%
Random Forest              87.8%
K-Nearest Neighbors        73.0%
Support Vector Machine     71.9%
Logistic Regression        73.9%
```

### Hyperparameter Optimization

```text
Optimization Technique    Best CV Score
----------------------------------------
GridSearchCV               87.1%
RandomizedSearchCV         86.9%
```

**Best Performing Model:** Random Forest Classifier

- Holdout Accuracy: 93%
- Average Cross-Validation Score: 87.8%
- GridSearchCV Score: 87.1%
- RandomizedSearchCV Score: 86.9%
# 💡 Key Insights
The dataset contains 768 patient records with eight medical and demographic input features.

The initial target distribution showed class imbalance, with 500 non-diabetic records and 268 diabetic records.

RandomOverSampler successfully balanced the dataset to 500 records in each class.

Data visualization using boxplots helped identify extreme values across numerical features.

Random Forest outperformed the other four classification algorithms in the model comparison.

Random Forest achieved an average cross-validation score of approximately 87.8%, demonstrating comparatively consistent performance across the evaluated folds.

Hyperparameter tuning using GridSearchCV and RandomizedSearchCV helped identify configurations with competitive cross-validation scores.

The project demonstrated the importance of data preprocessing, class balancing, algorithm comparison and model optimization in developing classification models.

# 🔗 Conclusion

This project demonstrates the application of Machine Learning techniques to a healthcare classification problem using a Diabetes Dataset.

Through data exploration, preprocessing, outlier handling, class balancing, model development, cross-validation and hyperparameter tuning, multiple classification algorithms were evaluated.

Among the five algorithms, Random Forest Classifier demonstrated the strongest performance, achieving a 93% holdout accuracy in the selected experiment and an average cross-validation score of approximately 87.8% in the model comparison.

The project provided practical experience in developing classification models, comparing algorithms, handling imbalanced datasets and optimizing model parameters.

### Future Improvements:

Prevent data leakage by applying oversampling only to training data.

Evaluate models using precision, recall, F1-score and ROC-AUC.

Perform feature importance analysis.

Explore advanced ensemble learning techniques.

Validate the final model on an independent test dataset.
