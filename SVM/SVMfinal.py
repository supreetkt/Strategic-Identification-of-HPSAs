
# coding: utf-8

# In[18]:


import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.utils import class_weight


# In[19]:


loaction='K:/Projects/ML/HPSA/master_final.csv'


# In[20]:


df=pd.read_csv(loaction)


# In[21]:


df.HPSA_Ind.value_counts()


# In[22]:


# For train test split
x_train=df.drop('HPSA_Ind', axis=1)
y_train = df.HPSA_Ind


# In[23]:


trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(x_train, y_train, test_size=0.3, random_state=24)


# In[24]:


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


# In[25]:


# For train test split
x_train = preprocess(trainFeatures)
x_test = preprocess(testFeatures)
y_train = trainLabels
y_test  = testLabels


# In[34]:


clf = SVC(C=1.5, kernel='rbf', degree=1, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
          class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)


# In[35]:


x_train_numpy = x_train.values
x_test_numpy = x_test.values
y_train_numpy = y_train.values
y_test_numpy = y_test.values


# In[36]:


cv = cross_val_score(clf, x_train_numpy, y_train_numpy, cv=10)


# In[37]:


cv


# In[38]:


np.mean(cv)


# In[39]:


clf.fit(x_train_numpy, y_train_numpy)


# In[40]:


predicted = clf.predict(x_test_numpy)


# In[41]:


accuracy_score(y_test_numpy, predicted)

