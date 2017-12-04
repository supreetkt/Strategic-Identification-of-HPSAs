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


 # Reading the csv file
location='K:/Projects/ML/HPSA/master_final.csv'
df=pd.read_csv(location)


# converting test class to 0 and 1
df['HPSA_Ind'].replace(axis=1,to_replace=1, value=0, inplace=True)
df['HPSA_Ind'].replace(axis=1,to_replace=2, value=1, inplace=True)


# For train test split
x_train=df.drop('HPSA_Ind', axis=1)
y_train = df.HPSA_Ind

trainFeatures, testFeatures, trainLabels, testLabels = train_test_split\
(x_train, y_train, test_size=0.3, random_state=24,stratify=y_train)


# preprocessing train set
le = LabelEncoder()

def preprocess(x_train):
        # segregating numeric and discrete values and dropping the columns where standard deviation is zero.
    x_train_numeric= x_train._get_numeric_data()
    x_train_dropstdzero=x_train_numeric.drop(x_train_numeric.loc[:, x_train_numeric.std()==0], axis=1) 
    col=x_train_dropstdzero.columns
	# imputing missing values with mean
    missingvalues=Imputer(missing_values=-1111.1, strategy='mean', axis=1, verbose=0, copy=False)   
    x_train_dropstdzero=missingvalues.fit_transform(x_train_dropstdzero)
    x_train_numeric=pd.DataFrame(data=x_train_dropstdzero,columns=col)
    
    # applying label encoder to discrete dataset
    x_train_discrete=x_train[['CHSI_State_Abbr']]
    x_train_discrete=x_train_discrete.apply(le.fit_transform)
	
    # standardising the train set by subtracting with mean and normalizing with standard deviation.
    standardised_x_train=(x_train_numeric-x_train_numeric.mean())/x_train_numeric.std()
    x_train=pd.concat([x_train_discrete.reset_index(drop=True),standardised_x_train],axis=1)
    #print(x_train.isnull().sum())
    return(x_train)


# For train test split
x_train = preprocess(trainFeatures)
x_test = preprocess(testFeatures)
y_train = trainLabels
y_test  = testLabels

# tuned parameters for SVM with rbf kernel

clfrbf=SVC(C=8,gamma=0.035625,kernel='rbf')

#  tuned parameters for linearSVM


clflinear=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=10000,
      penalty='l2', random_state=0, tol=0.0001,
     verbose=0)


#  tuned parameters for NuSVC with polynomial kernel


clfnusvc=NuSVC(cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.1, kernel='poly',
      max_iter=-1, nu=0.399, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)



# converting oandas dataframe to numpy array

x_train_numpy = x_train.values
x_test_numpy = x_test.values
y_train_numpy = y_train.values
y_test_numpy = y_test.values


# Training SVM, linearSVM and NuSVC models

clfrbf.fit(x_train_numpy, y_train_numpy)

clflinear.fit(x_train_numpy, y_train_numpy)

clfnusvc.fit(x_train_numpy, y_train_numpy)
 # prediction 
 
predictedrbf = clfrbf.predict(x_test_numpy)
predictedlinear = clflinear.predict(x_test_numpy)
predictednusvc = clfnusvc.predict(x_test_numpy)

# Accuracy for SVC with RBF kernel
accuracy_score(y_test_numpy, predictedrbf)

# Accuracy for LinearSVM
accuracy_score(y_test_numpy, predictedlinear)

#Accuracy for NuSVC
accuracy_score(y_test_numpy, predictednusvc)

confusionrbf = confusion_matrix(y_test_numpy, predictedrbf)
confusionlinear = confusion_matrix(y_test_numpy, predictedlinear)
confusionnusvc = confusion_matrix(y_test_numpy, predictednusvc)

# plotting roc curve

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


# accuracy measures for SVM with rbf

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

print(TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC)


# accuracy measures for LinearSVM


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

print(TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC)


# accuracy measures for NuSVC

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

print(TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC)