import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('../Preprocessing/master.csv', skipinitialspace=True)

target_arttibute = ['HPSA_Ind']
# 'State_FIPS_Code', 'County_FIPS_Code', 'CHSI_County_Name',	'CHSI_State_Name',
#              'CHSI_State_Abbr',	'Strata_ID_Number',	'No_Exercise', 'Obesity',
#              'High_Blood_Pres',	'Smoker	Diabetes',	'Uninsured', 'Elderly_Medicare',
#              'Disabled_Medicare', 'Prim_Care_Phys_Rate', 'Dentist_Rate', 'Community_Health_Center_Ind',
#              'HPSA_Ind', 'Influenzae', 'HepA', 'HepB', 'Measeles', 'Pertusis', 'Congential.Rubella',
#              'Syphilis', 'Unemployed', 'Sev_Work_Disabled', 'Major_Depression',
#              'Recent_Drug_Use',	'Ecol_Rpt', 'Ecol_Rpt_Ind', 'Ecol_Exp', 'Salm_Rpt',
#              'Salm_Rpt_Ind', 'Salm_Exp', 'Shig_Rpt', 'Shig_Rpt_Ind', 'Shig_Exp',
#              'Toxic_Chem', 'Carbon_Monoxide_Ind', 'Nitrogen_Dioxide_Ind',
#              'Sulfur_Dioxide_Ind', 'Ozone_Ind', 'Particulate_Matter_Ind	Lead_Ind',
#              'Premature	Infant_Mortality', 'Brst_Cancer', 'Col_Cancer', 'CHD',
#              'ALE',	'All_Death', 'Health_Status'

attribute = ['State_FIPS_Code', 'Strata_ID_Number',	'No_Exercise', 'Obesity',
             'High_Blood_Pres',	'Smoker	Diabetes',	'Uninsured', 'Elderly_Medicare',
             'Disabled_Medicare', 'Prim_Care_Phys_Rate', 'Dentist_Rate', 'Community_Health_Center_Ind',
             'Influenzae', 'HepA', 'HepB', 'Measeles', 'Pertusis', 'Congential.Rubella',
             'Syphilis', 'Unemployed', 'Sev_Work_Disabled', 'Major_Depression',
             'Recent_Drug_Use',	'Ecol_Rpt', 'Ecol_Rpt_Ind', 'Ecol_Exp', 'Salm_Rpt',
             'Salm_Rpt_Ind', 'Salm_Exp', 'Shig_Rpt', 'Shig_Rpt_Ind', 'Shig_Exp',
             'Toxic_Chem', 'Carbon_Monoxide_Ind', 'Nitrogen_Dioxide_Ind',
             'Sulfur_Dioxide_Ind', 'Ozone_Ind', 'Particulate_Matter_Ind	Lead_Ind',
             'Premature	Infant_Mortality', 'Brst_Cancer', 'Col_Cancer', 'CHD',
             'ALE',	'All_Death', 'Health_Status']

print(df[target_arttibute])
print(df[attribute])

# del df['id']
# del df['PlayerName']
# del df['sum_7yr_GP']
# del df['sum_7yr_TOI']
# del df['rs_PlusMinus']
# del df['Country']
#
# #convert into string values to int values
# #in position - convert to 0,1,2,3
# #CAN EURO USA - country
#
# mapping1 = {'CAN': 0, 'EURO': 1, 'USA':2}
# mapping2 = {'C':0, 'D':1, 'L':2, 'R':3}
# mapping3 = {'no':0, 'yes':1}
# df = df.replace({'country_group': mapping1, 'Position': mapping2, 'GP_greater_than_0':mapping3})
#
# training_set = df.loc[ df['DraftYear'] <= 2000]#np.array(training) if this doesn't work
# test_set = df.loc[df['DraftYear'] == 2001]
#
# train_classes = np.array(training_set['GP_greater_than_0'], dtype=int)
# test_classes = np.array(test_set['GP_greater_than_0'], dtype=int)
#
# gnb = GaussianNB()
#
# #train data-set and predict
# y_pred = gnb.fit(training_set, train_classes) #fit=creates a model which learns from this dataset, then predict using it.
#
# print(gnb.score(test_set, test_classes))