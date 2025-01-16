import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import  model_selection

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,  mean_squared_error
# database
from sklearn.datasets import load_boston

database = load_boston()

df = pd.DataFrame(data=database.data, columns=database.feature_names)
df['class'] = database.target
#df['class'] = df['class'].map({0:database.target_names[0], 1:database.target_names[1], 2:database.target_names[2]})
print(df.head(10), '\n')

print(df.describe(),'\n')

x = database.data
y = database.target
print(x.shape, y.shape, '\n')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print('Train database:', x_train.shape, y_train.shape)
print('Tests database:', x_test.shape, y_test.shape,'\n')

# HOLDOUT - 30/70
regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=30, min_samples_leaf=10, random_state=0)
regressor = regressor.fit(x_train, y_train.ravel()) #  Treina classificador

# Executa teste na base de testes
output_prediction = regressor.predict(x_test)

print("Resultado HOLDOUT 30/70")
print("R2 score: %.2f" % r2_score(y_test, output_prediction))
print("Mean squared error: %.2f" % mean_squared_error(y_test, output_prediction))

# Validação Cruzada
regressor2 = DecisionTreeRegressor(max_depth=5, min_samples_split=30, min_samples_leaf=10, random_state=0)
folds = 5
result = model_selection.cross_val_score(regressor2, x, y.ravel(), cv=folds)
print("Resultado Validacao Cruzada")
print("R2 scores = " + str(result))
print("R2 score médio: %.2f" % (result.mean()))