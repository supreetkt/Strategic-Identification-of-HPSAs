import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('../data/hockey.csv')
del df['id']
del df['PlayerName']
del df['sum_7yr_GP']
del df['sum_7yr_TOI']
del df['rs_PlusMinus']
del df['Country']

#convert into string values to int values
#in position - convert to 0,1,2,3
#CAN EURO USA - country

mapping1 = {'CAN': 0, 'EURO': 1, 'USA':2}
mapping2 = {'C':0, 'D':1, 'L':2, 'R':3}
mapping3 = {'no':0, 'yes':1}
df = df.replace({'country_group': mapping1, 'Position': mapping2, 'GP_greater_than_0':mapping3})

training_set = df.loc[ df['DraftYear'] <= 2000]#np.array(training) if this doesn't work
test_set = df.loc[df['DraftYear'] == 2001]

train_classes = np.array(training_set['GP_greater_than_0'], dtype=int)
test_classes = np.array(test_set['GP_greater_than_0'], dtype=int)

gnb = GaussianNB()

#train data-set and predict
y_pred = gnb.fit(training_set, train_classes) #fit=creates a model which learns from this dataset, then predict using it.

print(gnb.score(test_set, test_classes))