"""
This script demonstrates the use of a K-Nearest Neighbors (KNN) classifier to predict a target variable
based on a given dataset. It uses the Holdout method (70% training, 30% testing) for evaluation and calculates
performance metrics, including Kolmogorov-Smirnov (KS) statistics.

# Key Steps and Explanation:

1. **Load the Dataset:**
   - Reads data from a CSV file (`Resultado_Join_Campo__Menos_70_correlacao.csv`).
   - Separates the features (`x`) and target variable (`y`).
   - Splits the dataset into training and testing sets based on the "safra" column, with:
     - `TREINO`: Training data.
     - `TESTE`: Testing data.

2. **Data Preprocessing:**
   - Removes the "safra" column from both training and testing datasets, as it is not a feature for modeling.
   - Prints the shapes of the training and testing datasets for verification.

3. **Train the KNN Classifier:**
   - Initializes a KNN classifier with 3 neighbors (`n_neighbors=3`) and Euclidean distance (`p=2`).
   - Fits the model using the training data (`x_train`, `y_train`).

4. **Test the Classifier:**
   - Predicts target values for the test dataset (`x_test`).
   - Calculates the model's accuracy and confusion matrix on the test data.

5. **Calculate Kolmogorov-Smirnov (KS) Statistics:**
   - Computes the KS statistic and p-value to evaluate the model's ability to distinguish between classes.
   - Uses the probabilities predicted by the classifier (`predict_proba`).

6. **Performance Metrics:**
   - Calculates:
     - **Precision:** Proportion of true positives among all predicted positives.
     - **Recall (Sensitivity):** Proportion of true positives among all actual positives.
     - **F-measure:** Harmonic mean of precision and recall.
     - **Specificity:** Proportion of true negatives among all actual negatives.
   - Prints accuracy, KS index, confusion matrix, precision, and recall.

7. **Example Output:**
   - Accuracy and performance metrics for the KNN model.
   - Kolmogorov-Smirnov statistic and p-value.
   - Confusion matrix to evaluate classification results.

# Dependencies:
   - Libraries such as `pandas`, `numpy`, `scikit-learn`, and `scipy` are required. Ensure they are installed 
     in your Python environment.

# Example Use Case:
   - This script is suitable for binary or multi-class classification tasks where a dataset needs to be evaluated
     using the Holdout method.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import  model_selection
import scipy.stats as stats

# database
path = r".\Resultado_Join_Campo__Menos_70_correlacao.csv"
df = pd.read_csv(path, sep = ";")

x = df.drop(['target'], axis=1)
y = df.filter(['target'])

dfTreino = df[df["safra"] == 'TREINO']
dfTeste  = df[df["safra"] == 'TESTE']

# Holdout
# training
x_train = dfTreino.drop(['target'], axis=1)
y_train = dfTreino.filter(['target'])

# test
x_test = dfTeste.drop(['target'], axis=1)
y_test = dfTeste.filter(['target'])

#Remover coluna Safra
x_train = x_train.drop(['safra'], axis=1)
x_test = x_test.drop(['safra'], axis=1)

print('Train database:', x_train.shape, y_train.shape)
print('Tests database:', x_test.shape, y_test.shape,'\n')

# HOLDOUT - 30/70
classifier = KNeighborsClassifier(n_neighbors=3, p=2)
classifier = classifier.fit(x_train, y_train) #  Treina Classificador

# Executa teste na base de testes
predicted = classifier.predict(x_test)
score = classifier.score(x_test, y_test)
matrix = confusion_matrix(y_test, predicted)
print("Resultado HOLDOUT 30/70")

target_prob = classifier.predict_proba(x_test)

test = y_test['target'].to_numpy()
stats.ks_2samp(test, target_prob[:,1])

media = np.mean(test)
desvioPadrao = np.std(test, ddof=1)
ks_stat, ks_p_valor = stats.kstest(target_prob[:,1], cdf='norm', args = (media, desvioPadrao), N = len(target_prob[:,1]))

print(ks_stat)
print(ks_p_valor)

tp = matrix[0][0]
tn = matrix[1][1]
fp = matrix[1][0]
fn = matrix[0][1]

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f_measure = 2 / ( (1 / precision) + (1 / recall) )
especificidade = tn / (tn + fp)

print('\n\nTaxa de acerto: %.2f' % (score * 100) + "%")
print('Indice de Kolmogorov-Smirnov (KS): %.2f' % (ks_stat * 100) + "%" )
print("Matrix de confusão:")
print(matrix)
print('Precisão: %.2f' % (precision * 100) + "%" )
print('Revocação: %.2f' % (recall * 100) + "%" )
