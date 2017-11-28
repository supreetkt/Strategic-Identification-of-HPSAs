# Adaboost
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection

#1=No, 2=Yes in the dataset
def convert_target_label(x):
    if x == 1:
        return 0
    else:
        return 1

# Loading dataset: 42 columns
df = pd.read_csv(
    r'C:\Users\Guest123\Desktop\726 - Machine Learning\Project\ML-HPSA-Pandas\Preprocessed_Data\master_final.csv')
col_names = df.columns

#delete Nitrogen_Dioxide_Ind, Sulfur_Dioxide_Ind because they only have values 1s and don't affect the outcomes : 40 columns
del df['Nitrogen_Dioxide_Ind']
del df['Sulfur_Dioxide_Ind']

# converting discrete state values to continuous using dummy parameters : resultant df has 91 columns
df_state = pd.get_dummies(df['CHSI_State_Abbr'])
df = pd.concat([df, df_state], axis=1)
exclusion_list = list(df.CHSI_State_Abbr.unique()) + \
                  ['Carbon_Monoxide_Ind', 'Ozone_Ind', 'Particulate_Matter_Ind', 'Lead_Ind'] + \
                  ['Community_Health_Center_Ind']
del df['CHSI_State_Abbr']

#shuffling dataset : 90 columns
df = shuffle(df)

# selecting target label, converting columns from 1=No, 2=Yes to 0/1
HPSA_Ind_df = df['HPSA_Ind']
HPSA_Ind_df = HPSA_Ind_df.apply(convert_target_label)
df.Community_Health_Center_Ind = df.Community_Health_Center_Ind.apply(convert_target_label)
del df['HPSA_Ind'] #89 columns

#standardization: resultant = 89 columns
for i in df.columns:
    mean = np.mean(df[i])
    std = np.std(df[i])
    if std == 0:
        df.drop(i, axis=1, inplace=True)
    else:
        if i not in exclusion_list:
            df[i] = (df[i] - mean) / std

df = pd.concat([df, HPSA_Ind_df], axis=1) #90 columns

#apply algorithm - 1. Adaboost
array = df.values
X = array[:,0:89]
Y = array[:,89]
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
#dt = DecisionTreeClassifier()
model = AdaBoostClassifier(n_estimators=num_trees,learning_rate=1,random_state=seed) #base_estimator=dt,
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print('Accuracy of Adaboost modelling is: '+str(results.mean()*100) + '%')

#apply algorithm - 2.Gradient Boosting
array = df.values
X = array[:,0:89]
Y = array[:,89]
seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print('Accuracy of Gradient Boosting is: '+str(results.mean()*100) + '%')

#apply algorithm - 3. XG Boost
array = df.values
X = array[:,0:89]
Y = array[:,89]
seed = 7
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print('Accuracy of XG Boost is: '+str(accuracy*100) + '%')