import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

df = pd.read_csv('../Preprocessing/master_final.csv', skipinitialspace=True)

del df['Nitrogen_Dioxide_Ind']
del df['Sulfur_Dioxide_Ind']

# Required attributes
attributes = (df.columns).values.T.tolist()
attributes.remove('HPSA_Ind')
attributes.remove('CHSI_State_Abbr')

target_attribute = ['HPSA_Ind']

discrete_attributes = ['Carbon_Monoxide_Ind', 'Ozone_Ind', 'Particulate_Matter_Ind', 'Lead_Ind', 'Community_Health_Center_Ind']
continuous_attributes = list(set(attributes).difference(set(discrete_attributes)))

state_dummies = pd.get_dummies(df['CHSI_State_Abbr'])
df = pd.concat([df, state_dummies], axis=1)
del df['CHSI_State_Abbr']
#
dummy_attributes = list(set((df.columns).values.T.tolist()).difference(set(attributes)))
dummy_attributes.remove('HPSA_Ind')
# discrete_attributes = ['CHSI_State_Abbr']

df[target_attribute] = np.where(df[target_attribute] == 1, 0, 1)

df = shuffle(df)

discrete_attributes.extend(dummy_attributes)

train, test = train_test_split(df, test_size=0.3, random_state=24, stratify=df[target_attribute])

Y_train = np.array(train[target_attribute])
Y_test = np.array(test[target_attribute])

gaussian_model = GaussianNB()
multinomial_model = MultinomialNB()

trained_gaussian = gaussian_model.fit(np.array(train[continuous_attributes]), Y_train)
predicted_gaussian = trained_gaussian.predict_proba(test[continuous_attributes])

trained_multinomial = multinomial_model.fit(np.array(train[discrete_attributes]), Y_train)
predicted_multinomial = trained_multinomial.predict_proba(test[discrete_attributes])

predicted_proba = np.multiply(predicted_gaussian, predicted_multinomial)

Y_pred = []
for each in predicted_proba:
    if each[0] > each[1]:
        Y_pred.append(0)
    else:
        Y_pred.append(1)

accuracy = accuracy_score(Y_test, Y_pred, normalize=True)

# print(accuracy)

confusion = confusion_matrix(Y_test, Y_pred)
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
# y_score = model.predict_proba(X_test)[:, 1]
# print(y_score)
fpr, tpr, thresholds = roc_curve(Y_test, predicted_proba[:, 1])
area_under_curve = auc(fpr, tpr) * 100

print("----------------Naive Bayes----------------")
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

