
# coding: utf-8

# In[325]:


import numpy as np
from sklearn.svm import SVC,LinearSVC,NuSVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# In[293]:


loaction='K:/Projects/ML/HPSA/master_final.csv'


# In[294]:


df=pd.read_csv(loaction)


# In[295]:


df['HPSA_Ind'].replace(axis=1,to_replace=1, value=0, inplace=True)
df['HPSA_Ind'].replace(axis=1,to_replace=2, value=1, inplace=True)


# In[296]:


df.HPSA_Ind


# In[297]:


# df.to_csv('newMaster.csv', index=False)


# In[298]:


df.HPSA_Ind.value_counts()


# In[299]:


# For train test split
x_train=df.drop('HPSA_Ind', axis=1)
y_train = df.HPSA_Ind


# In[300]:


trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(x_train, y_train, test_size=0.3, random_state=24,stratify=y_train)


# In[303]:


# For preprocessing
le = LabelEncoder()

def preprocess(x_train):
    x_train_numeric= x_train._get_numeric_data()
    x_train_dropstdzero=x_train_numeric.drop(x_train_numeric.loc[:, x_train_numeric.std()==0], axis=1)
    col=x_train_dropstdzero.columns
    missingvalues=Imputer(missing_values=-1111.1, strategy='mean', axis=1, verbose=0, copy=False)
    x_train_dropstdzero=missingvalues.fit_transform(x_train_dropstdzero)
    x_train_numeric=pd.DataFrame(data=x_train_dropstdzero,columns=col)
    
    
    x_train_discrete=x_train[['CHSI_State_Abbr']]
    x_train_discrete=x_train_discrete.apply(le.fit_transform)
    
    standardised_x_train=(x_train_numeric-x_train_numeric.mean())/x_train_numeric.std()
    x_train=pd.concat([x_train_discrete.reset_index(drop=True),standardised_x_train],axis=1)
    #print(x_train.isnull().sum())
    return(x_train)


# In[304]:


# For train test split
x_train = preprocess(trainFeatures)
x_test = preprocess(testFeatures)
y_train = trainLabels
y_test  = testLabels


# In[305]:


# clf = SVC(C=1.5, kernel='rbf', degree=1, gamma=0.10000000000000001, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
#           class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)


# In[306]:



#clf=SVC(C=8,gamma=0.035625,kernel='rbf')


# In[307]:


# clf=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=10000,
#       penalty='l2', random_state=0, tol=0.0001,
#      verbose=0)


# In[494]:


clf=NuSVC(cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=2, gamma=0.1, kernel='poly',
      max_iter=-1, nu=0.399, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)


# In[495]:


x_train_numpy = x_train.values
x_test_numpy = x_test.values
y_train_numpy = y_train.values
y_test_numpy = y_test.values


# In[496]:


cv = cross_val_score(clf, x_train_numpy, y_train_numpy, cv=10)


# In[497]:


cv


# In[498]:


np.mean(cv)


# In[499]:


clf.fit(x_train_numpy, y_train_numpy)


# In[500]:


predicted = clf.predict(x_test_numpy)


# In[501]:


accuracy_score(y_test_numpy, predicted)


# In[502]:


confusion = confusion_matrix(y_test_numpy, predicted)
confusion


# In[503]:


# plot roc curve


# In[504]:


fpr, tpr, thresholds = roc_curve(y_test_numpy, np.array(predicted))
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='purple', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='coral', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[505]:


confusion.sum(axis=0)


# In[506]:


FP = confusion.sum(axis=0) - np.diag(confusion)  
FN = confusion.sum(axis=1) - np.diag(confusion)
TP = np.diag(confusion)
TN = confusion.sum() - (FP + FN + TP)


# In[507]:


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)


# In[508]:


TN = confusion[0,0]; FP = confusion[0,1]; FN = confusion[1,0]; TP = confusion[1,1];

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
NP = fn+tp # Num positive examples
NN = tn+fp # Num negative examples
Matrix  = NP+NN
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)


# In[509]:


ACC

