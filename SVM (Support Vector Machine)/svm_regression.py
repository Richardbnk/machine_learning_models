# This example loads the UCI Wine dataset, trains an SVM regressor using 
# the Holdout method, and another regressor using 10-fold cross-validation.

# Import necessary libraries
import numpy as np
import urllib
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the UCI Wine dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
raw_data = urllib.request.urlopen(url)

# Load the file as a matrix
dataset = np.loadtxt(raw_data, delimiter=",")

# Print the number of instances and attributes in the dataset
print(dataset.shape)

# Assign the 13 input attributes to X and the first attribute (class) to y as a regression target
# Note: In this example, we assume the first column can be treated as a continuous target for regression purposes
X = dataset[:, 1:13]
y = dataset[:, 0]

# EXAMPLE USING HOLDOUT
# Holdout -> splits the dataset into training (70%) and testing (30%), stratified
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the regressor
# Define the parameters to be evaluated during fine-tuning of the SVR
parameters = [
  {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
  {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']},
]

# Initialize the SVR regressor and perform hyperparameter tuning using GridSearchCV
regressor = SVR()
regressor = GridSearchCV(regressor, parameters, scoring='r2', cv=10, iid=False)
regressor = regressor.fit(X_train, y_train)
print("Best Parameters:", regressor.best_params_)

# Test the regressor using the test set
predicted = regressor.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)

# Display the results
print("\nHoldout Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# EXAMPLE USING CROSS-VALIDATION

# Initialize the SVR regressor and perform cross-validation
folds = 10
result = model_selection.cross_val_score(regressor, X, y, cv=folds, scoring='r2')
print("\nCross-Validation Results (%d folds):" % folds)
print(f"Mean R-squared: {result.mean():.2f}")
print(f"Standard Deviation: {result.std():.2f}")

# Perform cross-validation predictions
Z = model_selection.cross_val_predict(regressor, X, y, cv=folds)
mse_cv = mean_squared_error(y, Z)
r2_cv = r2_score(y, Z)

# Display cross-validation metrics
print("\nCross-Validation Metrics:")
print(f"Cross-Validation MSE: {mse_cv:.2f}")
print(f"Cross-Validation R2: {r2_cv:.2f}")
