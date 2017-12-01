
# coding: utf-8

# In[1]:


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


# In[2]:


loaction='K:/Projects/ML/HPSA/master_final.csv'


# In[3]:


df=pd.read_csv(loaction)


# In[4]:


df['HPSA_Ind'].replace(axis=1,to_replace=1, value=0, inplace=True)
df['HPSA_Ind'].replace(axis=1,to_replace=2, value=1, inplace=True)


# In[5]:


# df.to_csv('newMaster.csv', index=False)


# In[6]:


df.HPSA_Ind.value_counts()


# In[7]:


# For train test split
x_train=df.drop('HPSA_Ind', axis=1)
y_train = df.HPSA_Ind


# In[8]:


trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(x_train, y_train, test_size=0.3, random_state=24,stratify=y_train)


# In[9]:


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


# In[10]:


# For train test split
x_train = preprocess(trainFeatures)
x_test = preprocess(testFeatures)
y_train = trainLabels
y_test  = testLabels


# In[11]:


# clf = SVC(C=1.5, kernel='rbf', degree=1, gamma=0.10000000000000001, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
#           class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)


# In[12]:



clfrbf=SVC(C=8,gamma=0.035625,kernel='rbf')


# In[13]:


clflinear=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=10000,
      penalty='l2', random_state=0, tol=0.0001,
     verbose=0)


# In[14]:


clfnusvc=NuSVC(cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=2, gamma=0.1, kernel='poly',
      max_iter=-1, nu=0.399, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)


# In[15]:


x_train_numpy = x_train.values
x_test_numpy = x_test.values
y_train_numpy = y_train.values
y_test_numpy = y_test.values


# In[16]:


#cv = cross_val_score(clfrbf, x_train_numpy, y_train_numpy, cv=10)


# In[17]:


clfrbf.fit(x_train_numpy, y_train_numpy)


# In[18]:


clflinear.fit(x_train_numpy, y_train_numpy)


# In[19]:


clfnusvc.fit(x_train_numpy, y_train_numpy)


# In[20]:


predictedrbf = clfrbf.predict(x_test_numpy)


# In[21]:


predictedlinear = clflinear.predict(x_test_numpy)


# In[22]:


predictednusvc = clfnusvc.predict(x_test_numpy)


# In[23]:


# Accuracy for SVC with RBF kernel
accuracy_score(y_test_numpy, predictedrbf)


# In[24]:


# Accuracy for LinearSVC
accuracy_score(y_test_numpy, predictedlinear)


# In[25]:


#Accuracy for NuSVC
accuracy_score(y_test_numpy, predictednusvc)


# In[26]:


confusionrbf = confusion_matrix(y_test_numpy, predictedrbf)
confusionrbf


# In[27]:


confusionlinear = confusion_matrix(y_test_numpy, predictedlinear)
confusionlinear


# In[28]:


confusionnusvc = confusion_matrix(y_test_numpy, predictednusvc)
confusionnusvc


# In[29]:


# plot roc curve


# In[35]:


fpr, tpr, thresholds = roc_curve(y_test_numpy, np.array(predictedrbf))
fpr1, tpr1, thresholds = roc_curve(y_test_numpy, np.array(predictedlinear))
fpr2, tpr2, thresholds = roc_curve(y_test_numpy, np.array(predictednusvc))

roc_auc_rbf = auc(fpr, tpr)
roc_auc_linear = auc(fpr1, tpr1)
roc_auc_nusvc = auc(fpr2, tpr2)
plt.figure()
plt.plot(fpr, tpr, color='purple', label='ROC curve SVC RBF (area = %0.2f)' % roc_auc_rbf)
plt.plot(fpr1, tpr1, color='red', label='ROC curve LinearSVC (area = %0.2f)' % roc_auc_linear)
plt.plot(fpr2, tpr2, color='green', label='ROC curve NuSVC (area = %0.2f)' % roc_auc_nusvc)
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for SVM classifiers')
plt.legend(loc="lower right")
plt.show()


# In[31]:


thresholds


# In[32]:


TN = confusionrbf[0,0]; FP = confusionrbf[0,1]; FN = confusionrbf[1,0]; TP = confusionrbf[1,1];

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
NP =FN+TP # Num positive examples
NN = TN+FP # Num negative examples
Matrix  = NP+NN
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)


# In[33]:


TN = confusionlinear[0,0]; FP = confusionlinear[0,1]; FN = confusionlinear[1,0]; TP = confusionlinear[1,1];

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
NP =FN+TP # Num positive examples
NN = TN+FP # Num negative examples
Matrix  = NP+NN
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)


# In[34]:


TN = confusionnusvc[0,0]; FP = confusionnusvc[0,1]; FN = confusionnusvc[1,0]; TP = confusionnusvc[1,1];

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
NP =FN+TP # Num positive examples
NN = TN+FP # Num negative examples
Matrix  = NP+NN
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

