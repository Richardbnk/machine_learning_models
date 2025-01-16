
import numpy as np
import pandas as pd
import urllib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import  model_selection


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
raw_data = urllib.request.urlopen(url)

# Carrega arquivo como uma matriz
dataset = np.loadtxt(raw_data, delimiter=",")

# Imprime quantide de instancias e atributos da base
print("Instancias e atributos")
print(dataset.shape)

# Coloca em X os 13 atributos de entrada e em y as classes
# Observe que na base Wine a classe eh primeiro atributo 
x = dataset[:,1:10]
y = dataset[:,10:11]
print(x.shape, y.shape, '\n')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

print('Train database:', x_train.shape, y_train.shape)
print('Tests database:', x_test.shape, y_test.shape,'\n')

# HOLDOUT - 30/70
classifier = KNeighborsClassifier(n_neighbors=5, p=2)
classifier = classifier.fit(x_train, y_train.ravel()) #  Treina Classificador

# Executa teste na base de testes
predicted = classifier.predict(x_test)
score = classifier.score(x_test, y_test)
matrix = confusion_matrix(y_test, predicted)

print("Resultado HOLDOUT 30/70")
print("Accuracy = %.2f " % score, '\n')
print("Confusion Matrix:")
print(matrix)

# Validação Cruzada
classifier2 = KNeighborsClassifier(n_neighbors=5)
folds = 5
result = model_selection.cross_val_score(classifier2, x, y.ravel(), cv=folds)
print("\nCross Validation Results %d folds:" % folds)
print("Mean Accuracy: %.2f" % result.mean())
print("Mean Std: %.2f" % result.std())

# Matrix de Confusão
z = model_selection.cross_val_predict(classifier2, x, y.ravel(), cv=folds)
matrix = confusion_matrix(y, z)
print("Confusion Matrix:")
print(matrix)