# Este exemplo carrega a base Boston e treina uma Ã¡rvore de regressÃ£o usando
# holdout e outra usando validaÃ§Ã£o cruzada com 3 pastas. 

from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,  mean_squared_error
from sklearn import  model_selection
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO 
from sklearn.tree import export_graphviz
from IPython.display import Image  
from IPython.display import display
import pydotplus


# Carrega a base Boston
boston = load_boston()
X, y = boston['data'], boston['target']

print("PrediÃ§Ã£o do valor de residÃªncias em Boston (RegressÃ£o)")
print("Quantidade de instÃ¢ncias e atributos")
print(X.shape)

# Holdout 70% treinamento e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=0)

# Declara a Ã¡rvore de regressÃ£o
regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=30, min_samples_leaf=10, random_state=0)

# Treina a Ã¡rvore de regressÃ£o
regressor.fit(X_train, y_train)
output_prediction = regressor.predict(X_test)

# Imprime os resultados do experimento baseado em holdout
print("\nAvaliaÃ§Ã£o utilizando Holdout com 30% para teste")
print("R2 score: %.2f" % r2_score(y_test, output_prediction))

print("Mean squared error: %.2f" % mean_squared_error(y_test, output_prediction))

# ValidaÃ§Ã£o Cruzuzada com 3 folds
#declara uma nova Ã¡rvore de regressÃ£o
regressor1 = DecisionTreeRegressor(max_depth=5, min_samples_split=30, min_samples_leaf=10, random_state=0)

# define a quantidade de folds ou pastas e executa a validaÃ§Ã£o cruzada
folds=3
result = model_selection.cross_val_score(regressor1, X, y, cv=folds)

# Imprime os resultados da validaÃ§Ã£o cruzada
print("\nAvaliaÃ§Ã£o usando ValidaÃ§Ã£o Cruzada com %d folds" % folds)
print("R2 scores = " + str(result))
print("R2 score mÃ©dio: %.2f" % (result.mean()))

# Imprime a Ã¡rvore de regressÃ£o gerada
print("\nÃrvore Gerada no experimento baseado em Holdout")
dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
im=Image(graph.create_png())
display(im)
