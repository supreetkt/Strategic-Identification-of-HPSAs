
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[2]:


df = pd.read_csv('Downloads/master_final.csv')


# In[3]:


df['HPSA_Ind'].replace(axis=1,to_replace=1, value=0, inplace=True)
df['HPSA_Ind'].replace(axis=1,to_replace=2, value=1, inplace=True)


# In[4]:


# For train test split
x_train=df.drop('HPSA_Ind', axis=1)
y_train = df.HPSA_Ind


# In[5]:



trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(x_train, y_train, test_size=0.3, random_state=24,stratify=y_train)


# In[6]:


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


# In[7]:


# For train test split
x_train = preprocess(trainFeatures)
x_test = preprocess(testFeatures)
y_train = trainLabels
y_test  = testLabels


# In[8]:


x_train_numpy = x_train.values
x_test_numpy = x_test.values
y_train_numpy = y_train.values
y_test_numpy = y_test.values


# In[9]:


model = Sequential()
model.add(Dense(19,input_shape=(38,),kernel_initializer='uniform', bias_initializer='zeros',activation='sigmoid'))
#model.add(Dense(10,kernel_initializer='uniform',activation='sigmoid'))
model.add(Dense(1, kernel_initializer='uniform',activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(x_train_numpy, y_train_numpy, batch_size=5, nb_epoch=160)


# In[10]:


Y_pred = model.predict(x_test_numpy,batch_size=5)
score = model.evaluate(x_test_numpy, y_test_numpy, batch_size=5)
score


# In[11]:


predicted = list()
for i in np.nditer(Y_pred):
    if i>0.5:
        predicted.append(1)
    else:
        predicted.append(0)


# In[12]:


accuracy_score(y_test_numpy, predicted) # Accuracy


# In[14]:


accuracy_score(y_test_numpy, np.array(predicted)) # Accuracy
confusion_matrix(y_test_numpy, np.array(predicted)) # Confusion Matrix
precision_recall_fscore_support(y_test_numpy, np.array(predicted)) # Precision recall fscore and support for each class

fpr, tpr, thresholds = roc_curve(y_test_numpy, np.array(predicted))
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[18]:


confusion = confusion_matrix(y_test_numpy, predicted)

# FP = confusion.sum(axis=0) - np.diag(confusion)  
# FN = confusion.sum(axis=1) - np.diag(confusion)
# TP = np.diag(confusion)
# TN = confusion.sum() - (FP + FN + TP)


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
NP = FN+TP # Num positive examples
NN = TN+FP # Num negative examples
Matrix  = NP+NN
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)


# In[20]:


PPV

