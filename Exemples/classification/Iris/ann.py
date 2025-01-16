import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import  model_selection

from sklearn.neural_network import MLPClassifier
# database
from sklearn.datasets import load_iris

database = load_iris()

df = pd.DataFrame(data=database.data, columns=database.feature_names)
df['class'] = database.target
df['class'] = df['class'].map({0:database.target_names[0], 1:database.target_names[1], 2:database.target_names[2]})
print(df.head(10), '\n')

print(df.describe(),'\n')

x = database.data
y = database.target.reshape(-1, 1)
print(x.shape, y.shape, '\n')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

print('Train database:', x_train.shape, y_train.shape)
print('Tests database:', x_test.shape, y_test.shape,'\n')

# HOLDOUT - 30/70
classifier = MLPClassifier(hidden_layer_sizes=(8), activation='logistic', max_iter=1000,
                            solver='lbfgs', tol= 1e-10, verbose=True, early_stopping=True, 
                            validation_fraction=0.2)
classifier = classifier.fit(x_train, y_train.ravel()) #  Treina classificador

# Executa teste na base de testes
predicted = classifier.predict(x_test)
score = classifier.score(x_test, y_test)
matrix = confusion_matrix(y_test, predicted)

print("Resultado HOLDOUT 30/70")
print("Accuracy = %.2f " % score, '\n')
print("Confusion Matrix:")
print(matrix)

# Validação Cruzada
classifier2 = MLPClassifier(hidden_layer_sizes=(8), activation='logistic', max_iter=1000,
                            solver='lbfgs', tol= 1e-10, verbose=True, early_stopping=True, 
                            validation_fraction=0.2)
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

# Imprime a arvore gerada
#print("\nArvore gerada no experimento baseado em Holdout")
#dot_data = StringIO()
#export_graphviz(clfa, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True)
#
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#im=Image(graph.create_png())
#display(im)