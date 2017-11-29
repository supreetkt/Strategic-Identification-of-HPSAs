import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.utils import shuffle

df = pd.read_csv('../Preprocessing/master_final.csv', skipinitialspace=True)

del df['Nitrogen_Dioxide_Ind']
del df['Sulfur_Dioxide_Ind']

# Required attributes
attributes = (df.columns).values.T.tolist()
attributes.remove('HPSA_Ind')
attributes.remove('CHSI_State_Abbr')

target_attribute = ['HPSA_Ind']

binary_attributes = ['Carbon_Monoxide_Ind', 'Ozone_Ind', 'Particulate_Matter_Ind', 'Lead_Ind', 'Community_Health_Center_Ind']
continuous_attributes = list(set(attributes).difference(set(binary_attributes)))

state_dummies = pd.get_dummies(df['CHSI_State_Abbr'])
df = pd.concat([df, state_dummies], axis=1)
del df['CHSI_State_Abbr']
#
discrete_attributes = list(set((df.columns).values.T.tolist()).difference(set(attributes)))
discrete_attributes.remove('HPSA_Ind')
# discrete_attributes = ['CHSI_State_Abbr']

df[target_attribute] = np.where(df[target_attribute] == 1, 0, 1)

df = shuffle(df)

gaussian_model = GaussianNB()
multinomial_model = MultinomialNB()
bernoulli_model = BernoulliNB()

train, test = train_test_split(df, test_size=0.3)

train_class = np.array(train[target_attribute])
test_class = np.array(test[target_attribute])

trained_gaussian = gaussian_model.fit(np.array(train[continuous_attributes]), train_class)
predicted_gaussian = trained_gaussian.predict_proba(test[continuous_attributes])

trained_binary = bernoulli_model.fit(np.array(train[binary_attributes]), train_class)
predicted_bernoulli = trained_binary.predict_proba(test[binary_attributes])

trained_multinomial = multinomial_model.fit(np.array(train[discrete_attributes]), train_class)
predicted_multinomial = trained_multinomial.predict_proba(test[discrete_attributes])

predicted_proba = np.multiply(predicted_bernoulli, predicted_gaussian, predicted_multinomial)

predicted_class = []
for each in predicted_proba:
    if each[0] > each[1]:
        predicted_class.append(0)
    else:
        predicted_class.append(1)

accuracy = accuracy_score(test_class, predicted_class, normalize=True)

print(accuracy)

