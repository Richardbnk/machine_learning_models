import numpy as np
import urllib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils

# Carrega a base
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'
raw_data = urllib.request.urlopen(url)

# Carrega arquivo como uma matriz
dataset = np.loadtxt(raw_data, delimiter=",")

# Imprime quantide de instÃ¢ncias e atributos da base
print(dataset.shape)

# Coloca em X os 13 atributos de entrada e em y as classes
# Observe que na base Wine a classe Ã© primeiro atributo 
x = dataset[:,0:3]
y = dataset[:,3]

# EXEMPLO USANDO HOLDOUT
# Holdout -> dividindo a base em treinamento (70%) e teste (30%), estratificada
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3, random_state=42)

# declara o classificador
clfa = KNeighborsClassifier(n_neighbors=3)

#Alterar a label 
#lab_enc = preprocessing.LabelEncoder()
#training_scores_encoded = lab_enc.fit_transform(y_train)

# treina o classificador
clfa = clfa.fit(x_train, y_test.astype('int'))

# testa usando a base de testes
predicted=clfa.predict(x_test)

# calcula a acurÃ¡cia na base de teste
score=clfa.score(x_test, y_test)

# calcula a matriz de confusÃ£o
matrix = confusion_matrix(y_test, predicted)

# apresenta os resultados
print("Accuracy = %.2f " % score)
print("Confusion Matrix:")
print(matrix)

# EXEMPLO USANDO VALIDAÃ‡ÃƒO CRUZADA
clfb = KNeighborsClassifier(n_neighbors=3)
folds=10
result = model_selection.cross_val_score(clfb, x, y, cv=folds)
print("\nCross Validation Results %d folds:" % folds)
print("Mean Accuracy: %.2f" % result.mean())
print("Mean Std: %.2f" % result.std())

# matriz de confusÃ£o da validaÃ§Ã£o cruzada
z = model_selection.cross_val_predict(clfb, x, y, cv=folds)
cm = confusion_matrix(y, z)
print("Confusion Matrix:")
print(cm)
