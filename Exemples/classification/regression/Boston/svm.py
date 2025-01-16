import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import  model_selection

from sklearn.metrics import r2_score,  mean_squared_error

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
# database
from sklearn.datasets import load_boston

database = load_boston()

df = pd.DataFrame(data=database.data, columns=database.feature_names)
df['class'] = database.target
print(df.head(10), '\n')

print(df.describe(),'\n')

x = database.data
y = database.target
print(x.shape, y.shape, '\n')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print('Train database:', x_train.shape, y_train.shape)
print('Tests database:', x_test.shape, y_test.shape,'\n')

#Definicao dos parametros a serem avaliados no ajuste fino do SVM
parameters = [
  {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'kernel': ['linerar']},
  {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'gamma': [0.1, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']},
 ]

regressor = SVR()
regressor = GridSearchCV(regressor, parameters, n_jobs=8, verbose=2)
regressor = regressor.fit(x_train, y_train)
print(regressor.best_params_)

# Executa teste na base de testes
output_prediction = regressor.predict(x_test)

print("Resultado HOLDOUT 30/70")
print("R2 score: %.2f" % r2_score(y_test, output_prediction))
print("Mean squared error: %.2f" % mean_squared_error(y_test, output_prediction))

# Validação Cruzada
regressor2 = SVR()
folds = 5
result = model_selection.cross_val_score(regressor2, x, y.ravel(), cv=folds)
print("Resultado Validacao Cruzada")
print("R2 scores = " + str(result))
print("R2 score médio: %.2f" % (result.mean()))

# Imprime a arvore gerada
#print("\nArvore gerada no experimento baseado em Holdout")
#dot_data = StringIO()
#export_graphviz(regressor, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True)
#
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#im=Image(graph.create_png())
#display(im)