"""
This script demonstrates the use of the Naive Bayes classifier (GaussianNB) for predicting a target variable
based on a dataset. The Holdout method (70% training, 30% testing) is used for model evaluation, and 
performance metrics such as Kolmogorov-Smirnov (KS) statistics, precision, and recall are calculated.

# Key Steps and Explanation:

1. **Load the Dataset:**
   - Reads data from a CSV file (`Resultado_Join_Campo__Menos_70_correlacao.csv`).
   - Separates the features (`x`) and target variable (`y`).
   - Splits the dataset into training (`TREINO`) and testing (`TESTE`) subsets based on the "safra" column.

2. **Data Preprocessing:**
   - Removes the "safra" column, as it is not used for training or testing.
   - Prints the shapes of the training and testing datasets for verification.

3. **Train the Naive Bayes Classifier:**
   - Initializes a Gaussian Naive Bayes classifier (`GaussianNB`).
   - Trains the model using the training data (`x_train`, `y_train`).

4. **Test the Classifier:**
   - Predicts the target values for the test dataset (`x_test`).
   - Computes the accuracy score and confusion matrix on the test data.

5. **Kolmogorov-Smirnov (KS) Statistic:**
   - Evaluates the model's ability to distinguish between classes using predicted probabilities.
   - Calculates the KS statistic and p-value using the test data and predicted probabilities.

6. **Performance Metrics:**
   - Calculates:
     - **Precision:** Ratio of true positives to predicted positives.
     - **Recall (Sensitivity):** Ratio of true positives to actual positives.
     - **F-measure:** Harmonic mean of precision and recall.
     - **Specificity:** Ratio of true negatives to actual negatives.
   - Prints accuracy, KS index, confusion matrix, precision, recall, and other metrics.

7. **Example Output:**
   - Training and testing dataset shapes.
   - Accuracy score, KS statistic, confusion matrix, and key metrics for the classifier.

# Dependencies:
   - Libraries such as `pandas`, `numpy`, `scikit-learn`, and `scipy` are required.
   - Install them using pip if necessary:
     ```bash
     pip install pandas numpy scikit-learn scipy
     ```

# Example Use Case:
   - This script is ideal for binary or multi-class classification tasks where you need a probabilistic model 
     to predict class membership.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import  model_selection
import scipy.stats as stats

# database
path = r"D:\1. Dados\11. PUC\Machine Learning para Big Data\Aula 4 - ETL e Analise do dados\Resultado_Join_Campo__Menos_70_correlacao.csv"
df = pd.read_csv(path, sep = ";")

x = df.drop(['target'], axis=1)
y = df.filter(['target'])

dfTreino = df[df["safra"] == 'TREINO']
dfTeste  = df[df["safra"] == 'TESTE']

# Holdout
#Treinamento
x_train = dfTreino.drop(['target'], axis=1)
y_train = dfTreino.filter(['target'])

#Teste
x_test = dfTeste.drop(['target'], axis=1)
y_test = dfTeste.filter(['target'])

#Remover coluna Safra
x_train = x_train.drop(['safra'], axis=1)
x_test = x_test.drop(['safra'], axis=1)

print('Train database:', x_train.shape, y_train.shape)
print('Tests database:', x_test.shape, y_test.shape,'\n')

# HOLDOUT - 30/70
classifier = GaussianNB()
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
