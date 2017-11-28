# AdaBoost Classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas
from sklearn import model_selection

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
dt = DecisionTreeClassifier()
model = AdaBoostClassifier(n_estimators=num_trees, base_estimator=dt,learning_rate=1 ,random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())