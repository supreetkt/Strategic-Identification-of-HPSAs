# XGBoost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc, roc_curve

#1=No, 2=Yes in the dataset
def convert_target_label(x):
    if x == 1:
        return 0
    else:
        return 1

# Loading dataset: 42 columns
df = pd.read_csv(
    r'C:\Users\Guest123\Desktop\726 - Machine Learning\Project\ML-HPSA-Pandas\Preprocessing\master_final.csv')
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

#define datasets and class labels X, Y
array = df.values
X = array[:,0:89]
Y = array[:,89]

#split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify = Y)

#apply algorithm - 3. XGBoost
xgboostt = XGBClassifier()#AdaBoostClassifier(n_estimators=num_trees, learning_rate=1, random_state = seed)
cross = cross_val_score(xgboostt, X_train, Y_train, cv=10)
xgboostt.fit(X_train, Y_train)
y_pred = xgboostt.predict(X_test)

# confusion matrix
confusion = confusion_matrix(Y_test, y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

#Classification Accuracy
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)*100 #accuracy_score(Y, y_pred)*100

#Misclassification Rate/Classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)*100

#Sensitivity/True Positive Rate/Recall
sensitivity = TP / float(FN + TP)*100 #or recall_score

#Specificity/True Negative Rate
specificity = TN / float(TN + FP)*100

#False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
false_positive_rate = FP / float(TN + FP)*100

#False Negative Rate: When the actual value is positive, how often is the prediction incorrect?
false_negative_rate = FN / float(FN + TP)*100

#Precision
precision = TP / float(TP + FP)*100

#aoc, roc
#store the predicted probabilities for class 1
y_score = xgboostt.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, y_score)
area_under_curve = auc(fpr, tpr) * 100

print('------------------Metrics------------------')
print('XGBoost------------------------------------')
print('Confusion matrix (TN, FP, FN, TP): (' +str(TN) + ', ' + str(FP) + ', ' + str(FN) + ', ' + str(TP) + ')')
print('Classification Accuracy = %.2f %%' %classification_accuracy)
print('Classification Error = %.2f %%' % classification_error)
print('False Positive Rate = %.2f %%' % false_positive_rate)
print('False Negative Rate = %.2f %%'% false_negative_rate)
print('True Positive Rate/Sensitivity/Recall = %.2f %%'% sensitivity)

print('Precision = %.2f %%' % precision)
print('Specificity = %.2f %%' % specificity)
print('Area under curve = %.2f %%' % area_under_curve)

print('\n\n\n-------------------Plot-------------------')
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for HPSA classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()