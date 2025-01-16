# This example loads the UCI Wine dataset, trains an MLP classifier using 
# Holdout and another using 10-fold cross-validation.

# Import necessary libraries
import numpy as np
import urllib
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Load the UCI Wine dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
raw_data = urllib.request.urlopen(url)

# Load the file as a matrix
dataset = np.loadtxt(raw_data, delimiter=",")

# Print the number of instances and attributes in the dataset
print(dataset.shape)

# Assign the 13 input attributes to X and the classes to y
# Note that in the Wine dataset, the class is the first attribute
X = dataset[:, 1:13]
y = dataset[:, 0]

# EXAMPLE USING HOLDOUT
# Holdout -> split the dataset into training (70%) and testing (30%) sets, stratified
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train the classifier
clfa = MLPClassifier(hidden_layer_sizes=(8), activation='logistic', max_iter=1000,
                      solver='lbfgs', tol=1e-10, verbose=True, early_stopping=True, validation_fraction=0.2)
clfa = clfa.fit(X_train, y_train)

# Test using the test set
predicted = clfa.predict(X_test)

# Calculate the accuracy on the test set
score = clfa.score(X_test, y_test)

# Calculate the confusion matrix
matrix = confusion_matrix(y_test, predicted)

# Display the results
print("Accuracy = %.2f " % score)
print("Confusion Matrix:")
print(matrix)

# EXAMPLE USING CROSS-VALIDATION
clfb = MLPClassifier(hidden_layer_sizes=(8), activation='logistic', max_iter=1000,
                      solver='lbfgs', tol=1e-10, verbose=True, early_stopping=True, validation_fraction=0.2)
folds = 10
result = model_selection.cross_val_score(clfb, X, y, cv=folds)
print("\nCross-Validation Results (%d folds):" % folds)
print("Mean Accuracy: %.2f" % result.mean())
print("Standard Deviation: %.2f" % result.std())

# Confusion matrix for cross-validation
Z = model_selection.cross_val_predict(clfb, X, y, cv=folds)
cm = confusion_matrix(y, Z)
print("Confusion Matrix:")
print(cm)
