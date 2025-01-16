import numpy as np
import pandas as pd
import urllib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import  model_selection

from sklearn.tree import DecisionTreeClassifier

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
raw_data = urllib.request.urlopen(url)

# Carrega arquivo como uma matriz
dataset = np.loadtxt(raw_data, delimiter=",")

# Imprime quantide de instancias e atributos da base
print("Instancias e atributos")
print(dataset.shape)

# Coloca em X os 13 atributos de entrada e em y as classes
# Observe que na base Wine a classe eh primeiro atributo 
x = dataset[:,0:3]
y = dataset[:,3:4]

print(x.shape, y.shape, '\n')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

print('Train database:', x_train.shape, y_train.shape)
print('Tests database:', x_test.shape, y_test.shape,'\n')

# HOLDOUT - 30/70
classifier = DecisionTreeClassifier(criterion='entropy')
classifier = classifier.fit(x_train, y_train.ravel()) #  Treina classificador

# Executa teste na base de testes
predicted = classifier.predict(x_test)
score = classifier.score(x_test, y_test)
matrix = confusion_matrix(y_test, predicted)

tp = matrix[0][0]
tn = matrix[1][1]
fp = matrix[1][0]
fn = matrix[0][1]

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f_measure = 2 / ( (1 / precision) + (1 / recall) )
especificidade = tn / (tn + fp)

print('\n\nTaxa de acerto: %.2f' % (score * 100) + "%")
print('F1: %.2f' % (f_measure * 100) + "%" )
print('Precisão: %.2f' % (precision * 100) + "%" )
print('Revocação: %.2f' % (recall * 100) + "%" )
print('Especificidade: %.2f' % (especificidade * 100) + "%" )
print("Matrix de confusão:")
print(matrix)