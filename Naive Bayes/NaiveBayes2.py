import pandas as pd
import numpy as np
import math

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

discrete_attributes = ['Carbon_Monoxide_Ind', 'Ozone_Ind', 'Particulate_Matter_Ind', 'Lead_Ind', 'Community_Health_Center_Ind', 'CHSI_State_Abbr']
continuous_attributes = list(set(attributes).difference(set(discrete_attributes)))
df[target_attribute] = np.where(df[target_attribute] == 1, 0, 1)


def get_priors(data):
    count_yes = data[target_attribute][data[target_attribute] == 1].count()
    count_no = data[target_attribute][data[target_attribute] == 0].count()
    total_count = data[target_attribute].count()
    prior_yes = count_yes / (total_count * 1.0)
    prior_no = count_no / (total_count * 1.0)
    return prior_yes, prior_no


def get_rows_for_classes(data, att):
    class_true = data.loc[data[target_attribute[0]] == 1, att]
    class_false = data.loc[data[target_attribute[0]] == 0, att]
    return class_true, class_false


def get_prob_discrete(data, attr):
    prob_dict = {}
    unique_row = data[attr].unique()

    for each_row in unique_row:
        count = len(data.loc[data[attr] == each_row, attr])
        prob_dict[each_row] = count / len(data)

    return prob_dict


def get_mean_var(dataset, att):
    mean = dataset[att].mean()
    variance = dataset[att].var(ddof=0)
    return mean, variance


def calculate_probability_each(x, mean, var):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var)))
    prob = (1 / math.sqrt(2 * math.pi * var)) * exponent
    return prob


def get_accuracy(results, data):
    count = 0
    for eachTestIndex in range(0, len(data)):
        if results[eachTestIndex] == data[target_attribute].values[eachTestIndex]:
            count = count + 1
    return count / (len(data) * 1.0)


train, test = train_test_split(df, test_size=0.3, random_state=24
                               , stratify=df[target_attribute])

# Calculate Priors on training data
prob_prior_yes, prob_prior_no = get_priors(train)

# Get rows for Yes and No result from training dataset
yes_class, no_class = get_rows_for_classes(train, continuous_attributes)

# Get rows for Yes and No result from training dataset for discrete.
discrete_yes, discrete_no = get_rows_for_classes(train, discrete_attributes)

# Get Probability of each country and position in a map(dictionary)
prob_discrete_yes = {}
prob_discrete_no = {}

for each_attribute in discrete_attributes:
    prob_discrete_yes[each_attribute] = get_prob_discrete(discrete_yes, each_attribute)
    prob_discrete_no[each_attribute] = get_prob_discrete(discrete_no, each_attribute)

# Mean and variance of each feature by class = "yes" on training data
mean_yes, variance_yes = get_mean_var(yes_class, continuous_attributes)

# Mean and variance of each feature by class = "no" on training data
mean_no, variance_no = get_mean_var(no_class, continuous_attributes)

# ====== Use test data ============
# Result of classification on test data
result_test_data = []



# Loop through each of test data and classify
for i in range(0, len(test)):
    eachTestData_gau = test[continuous_attributes].values[i]
    eachTestData_dis = test[discrete_attributes].values[i]

    # Find class for test data
    # We can ignore the marginal probability (the denominator) since it will be same for all
    # We are actually calculating is this:
    # numerator_of_posterior_yes = prob_prior_yes * prob (each attr / yes)
    # numerator_of_posterior_no = prob_prior_no * prob (each attr / no)
    numerator_of_posterior_yes = prob_prior_yes
    numerator_of_posterior_no = prob_prior_no

    for j in range(0, len(eachTestData_gau)):
        numerator_of_posterior_yes = numerator_of_posterior_yes * calculate_probability_each(
                                        eachTestData_gau[j],
                                        mean_yes[j],
                                        variance_yes[j])
        numerator_of_posterior_no = numerator_of_posterior_no * calculate_probability_each(
                                        eachTestData_gau[j], mean_no[j],
                                        variance_no[j])

    # multiplying posterior of discrete datas.
    for a in range(len(prob_discrete_yes)):
        for b in range(len(eachTestData_dis)):
            if a == b:
                multi = prob_discrete_yes.get(discrete_attributes[a]).get(eachTestData_dis[b])
                if multi is not None:
                    numerator_of_posterior_yes = numerator_of_posterior_yes * multi

    for a in range(len(prob_discrete_no)):
        for b in range(len(eachTestData_dis)):
            if a == b:
                multi = prob_discrete_no.get(discrete_attributes[a]).get(eachTestData_dis[b])
                if multi is not None:
                    numerator_of_posterior_no = numerator_of_posterior_no * multi

    # Which ever is greater, we classify the test instance to that class.
    if numerator_of_posterior_yes[0] >= numerator_of_posterior_no[0]:
        result_test_data.append(1)
    else:
        result_test_data.append(0)

print('Accuracy:', get_accuracy(result_test_data, test) * 100, "%")


