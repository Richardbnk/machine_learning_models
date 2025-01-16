import numpy as np
import pandas as pd
import urllib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import  model_selection

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# database

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

#Definicao dos parametros a serem avaliados no ajuste fino do SVM
parameters = [
  {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'kernel': ['linear']},
  {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'gamma': [0.1, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']},
 ]

classifier = SVC()
classifier = GridSearchCV(classifier, parameters, scoring = 'accuracy', cv=10, iid=False)
classifier = classifier.fit(x_train, y_train)
print(classifier.best_params_)


# Executa testes na base de testes
predicted = classifier.predict(x_test)
score = classifier.score(x_test, y_test)
matrix = confusion_matrix(y_test, predicted)

print("Resultado HOLDOUT 30/70")
print("Accuracy = %.2f " % score, '\n')
print("Confusion Matrix:")
print(matrix)

# Validação Cruzada
classifier2 = SVC()
classifier2 = GridSearchCV(classifier2, parameters, scoring = 'accuracy', cv=10, iid=False)
folds=10
result = model_selection.cross_val_score(classifier2, x, y, cv=folds)
print("\nCross Validation Results %d folds:" % folds)
print("Mean Accuracy: %.2f" % result.mean())
print("Mean Std: %.2f" % result.std())

# Matrix de Confusão
z = model_selection.cross_val_predict(classifier2, x, y, cv=folds)
matrix = confusion_matrix(y, z)
print("Confusion Matrix:")
print(matrix)

# Imprime a arvore gerada
#print("\nArvore gerada no experimento baseado em Holdout")
#dot_data = StringIO()
#export_graphviz(classifier, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True)
#
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#im=Image(graph.create_png())
#display(im)