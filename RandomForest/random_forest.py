from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

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

# For train test split
HPSA_data = np.array(df.drop('HPSA_Ind', axis=1))
HPSA_target = np.array(df.HPSA_Ind)

cv_scores = []
cv_preds = []

X_train, X_test, Y_train, Y_test = train_test_split(HPSA_data, HPSA_target,
                                                    test_size=0.3,
                                                    random_state=24,
                                                    stratify=HPSA_target)

rfc_model = RandomForestClassifier(random_state=50)
param_grid = { "n_estimators": [200, 300],
           "criterion": ["gini", "entropy"],
           "max_features": ["auto", "sqrt"],
           "max_depth": [10, 20] }

grid_search = GridSearchCV(rfc_model, param_grid, n_jobs=-1, cv=2)
grid_search.fit(X_train, Y_train)
print (grid_search.best_params_)

# Optimized RF classifier
rfc = RandomForestClassifier(**grid_search.best_params_)


kfold = model_selection.KFold(n_splits=10, random_state=123)

# fit the model with training set
scores = model_selection.cross_val_score(rfc, X_train, Y_train, cv=kfold,
                                         scoring='accuracy')
cv_scores.append(scores.mean() * 100)
print("Train accuracy %0.2f (+/- %0.2f)" % (
scores.mean() * 100, scores.std() * 100))

# predict on testing set
Y_preds = model_selection.cross_val_predict(rfc, X_test, Y_test, cv=kfold)
cv_preds.append(accuracy_score(Y_test, Y_preds) * 100)
# print("Test accuracy %0.2f" % (100 * accuracy_score(Y_test, Y_preds)))

# model = RandomForestClassifier(max_depth=20,
#                                n_estimators=200,
#                                n_jobs=-1,
#                                random_state=50,
#                                max_features="auto")
rfc.fit(X_train, Y_train)
#
# # score = model.score(X_test, Y_test)
# # print(score)
#
# y_pred = model.predict(X_test)
# print ("-----------------------------------------------")
# print(y_pred)
#
confusion = confusion_matrix(Y_test, Y_preds)
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

#Specificity
specificity = TN / float(TN + FP)*100

#False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
false_positive_rate = FP / float(TN + FP)*100

#False Negative Rate: When the actual value is positive, how often is the prediction incorrect?
false_negative_rate = FN / float(FN + TP) * 100

#Precision
precision = TP / float(TP + FP)*100

#aoc, roc
#store the predicted probabilities for class 1
# proba = model_selection.cross_val_predict(rfc, X_test, Y_test, cv=kfold, method='predict_proba')
Y_score = rfc.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_score)
area_under_curve = auc(fpr, tpr) * 100

print("----------------Random Forest----------------")
print('Confusion matrix (TN, FP, FN, TP)= (' +str(TN) + ', ' + str(FP) + ', ' + str(FN) + ', ' + str(TP) + ')')
print("Classification Accuracy = ", classification_accuracy)
print("Classification Error = ", classification_error)
print("False Positive Rate = ", false_positive_rate)
print("False Negative Rate = ", false_negative_rate)
print("True Positive Rate/Sensitivity/Recall = ", sensitivity)
print("Precision = ", precision)
print("Specificity = ", specificity)
print("Area under curve = ", area_under_curve)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for HPSA classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
