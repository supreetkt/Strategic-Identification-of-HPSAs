from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import pandas as pd
import numpy as np

# Read Preprocessed Data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

df = pd.read_csv('../Preprocessing/master_final.csv', skipinitialspace=True)

# Remove two columns since it has only 1 value
del df['Nitrogen_Dioxide_Ind']
del df['Sulfur_Dioxide_Ind']

# Required attributes
attributes = (df.columns).values.T.tolist()
attributes.remove('HPSA_Ind')
attributes.remove('CHSI_State_Abbr')

target_attribute = ['HPSA_Ind']
df[target_attribute] = np.where(df[target_attribute] == 1, 0, 1)

# TODO: Standardization of continuous attributes
binary_attributes = ['Carbon_Monoxide_Ind', 'Ozone_Ind', 'Particulate_Matter_Ind', 'Lead_Ind', 'Community_Health_Center_Ind']
continuous_attributes = list(set(attributes).difference(set(binary_attributes)))

# Get Mean
mean = df[continuous_attributes].mean()
# get standard Deviaton
standard_deviation = df[continuous_attributes].std(ddof=0)

# Subtract the mean of the column from each entry and
# Divide each entry by the standard deviation of the column
for i in range(len(continuous_attributes)):
    df[continuous_attributes[i]] = (df[continuous_attributes[i]] - mean[i]) / \
                                   standard_deviation[i]

# Change discrete into dummy variables and concatenating to dataframe
state_dummies = pd.get_dummies(df['CHSI_State_Abbr'])
df = pd.concat([df, state_dummies], axis=1)
del df['CHSI_State_Abbr']

attributes = (df.columns).values.T.tolist()
attributes.remove('HPSA_Ind')

df = shuffle(df)

train, test = train_test_split(df, test_size=0.3)

X_train = np.array(train[attributes])
X_test = np.array(test[attributes])

Y_train = np.array(train[target_attribute])
Y_test = np.array(test[target_attribute])

# TODO: Take Train and Test Data anf fit into model

model = RandomForestClassifier(max_depth=20, random_state=0)
model.fit(X_train, Y_train)

# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=2, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#             oob_score=False, random_state=0, verbose=0, warm_start=False)

# print(clf.feature_importances_)
# [ 0.17287856  0.80608704  0.01884792  0.00218648]
# print(clf.predict([[0, 0, 0, 0]]))

score = model.score(X_test, Y_test)

print(score)