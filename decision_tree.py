"""
This example demonstrates how to load the Wine dataset from UCI, train a Decision Tree using 
the Holdout method, and validate it using 10-fold cross-validation.

# Key Steps and Explanation:

1. **Import Required Libraries:**
   - Includes libraries like NumPy, pandas, scikit-learn, and SciPy for data handling, modeling, 
     and statistical evaluation.

2. **Load the Dataset:**
   - The dataset is loaded from a specified CSV file (`Resultado_Join_Campo__Menos_70_correlacao.csv`).
   - Features (`x`) and the target variable (`y`) are separated.

3. **Split Data into Training and Testing Sets:**
   - The dataset is split based on the "safra" column, which indicates training ("TREINO") 
     and testing ("TESTE") data.

4. **Prepare Data for Holdout Validation:**
   - `x_train` and `x_test` represent the feature sets for training and testing, respectively.
   - `y_train` and `y_test` are the corresponding target labels.
   - The "safra" column is dropped as it is not a feature for the model.

5. **Train a Decision Tree Classifier:**
   - A decision tree classifier is declared and trained on the training dataset using entropy as the criterion.
   - The classifier predicts the target values for the test dataset.

6. **Evaluate the Model (Holdout Results):**
   - Calculates key performance metrics, including:
     - Accuracy (`score`): Overall success rate of the classifier.
     - Confusion Matrix (`matrix`): Details true/false positives and negatives.
     - Kolmogorov-Smirnov (KS) Statistic (`ks_stat`): Evaluates the model's ability to distinguish between classes.

7. **Calculate Performance Metrics:**
   - Precision: How many predicted positives are actually correct.
   - Recall (Sensitivity): How many actual positives are correctly predicted.
   - F-measure: Harmonic mean of precision and recall.
   - Specificity: How well the model identifies negatives.

8. **Print Results:**
   - Displays accuracy, KS index, confusion matrix, precision, recall, and other key statistics.

# Example Output:
   - Training and testing dataset shapes.
   - Accuracy score for the model.
   - Confusion matrix with predictions.
   - KS statistics and performance metrics.

# Dependencies:
   - Libraries such as `pandas`, `numpy`, `scikit-learn`, and `scipy` are required. Ensure they are installed 
     in your environment.

# Use Case:
   - This script is suitable for evaluating a decision tree classifier on a dataset with a defined training 
     and testing split. It highlights key evaluation metrics that are crucial for assessing the model's performance.
"""

# Importa bibliotecas necessarias 
import numpy as np
import urllib
from sklearn import tree
from sklearn import  model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz
from IPython.display import Image  
from IPython.display import display
import pydotplus

import pandas as pd

import scipy.stats as stats

path = r".\Resultado_Join_Campo__Menos_70_correlacao.csv"
df = pd.read_csv(path, sep = ";")

x = df.drop(['target'], axis=1)
y = df.filter(['target'])

dfTreino = df[df["safra"] == 'TREINO']
dfTeste  = df[df["safra"] == 'TESTE']

# Holdout
# Training
x_train = dfTreino.drop(['target'], axis=1)
y_train = dfTreino.filter(['target'])

# Test
x_test = dfTeste.drop(['target'], axis=1)
y_test = dfTeste.filter(['target'])

# remove unnecessary columns
x_train = x_train.drop(['safra'], axis=1)
x_test = x_test.drop(['safra'], axis=1)

print('Train database:', x_train.shape, y_train.shape)
print('Tests database:', x_test.shape, y_test.shape,'\n')

# classificador
classifier = tree.DecisionTreeClassifier(criterion='entropy')

# train classificador
classifier = classifier.fit(x_train, y_train)

# predict test dataset
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
