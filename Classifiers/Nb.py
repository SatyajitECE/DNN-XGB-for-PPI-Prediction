import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

df_pos= pd.read_csv('PositiveYH.csv',header=None)
df_neg = pd.read_csv('NegativeYH.csv',header=None)


df_neg['Status'] = 0
df_pos['Status'] = 1
df_neg=df_neg.sample(n=len(df_pos))

df = pd.concat([df_pos,df_neg])
df = df.reset_index()
df=df.sample(frac=1)
df = df.iloc[:,1:]

X = df.iloc[:,0:1986].values
y = df.iloc[:,1986:].values

scaler=RobustScaler()
X=scaler.fit_transform(X)
kf=StratifiedKFold(n_splits=5)

accuracy = []
specificity = []
sensitivity = []
precision=[]
recall=[]
m_coef=[]

auc_list=[]
Nb_fpr_list=[]
Nb_tpr_list=[]
o=0
max_accuracy=float("-inf")
Nb_fpr=None
Nb_tpr=None

for train, test in kf.split(X,y):
    o=o+1
    cv_clf=GaussianNB()
    y_train=np_utils.to_categorical(y[train])
    hist=cv_clf.fit(X[train], y[train])
    y_score=cv_clf.predict_proba(X[test])
    y_test=np_utils.to_categorical(y[test])  
    
    fpr, tpr, _ = roc_curve(y_test[:,0].ravel(), y_score[:,0].ravel())
    auc = metrics.roc_auc_score(y_test, y_score)
    auc_list.append(auc)
    coef=matthews_corrcoef(y_test.argmax(axis=1), y_score.argmax(axis=1), sample_weight=None)
    m_coef.append(coef)
    
    cm1=confusion_matrix(y_test.argmax(axis=1), y_score.argmax(axis=1))
    acc = (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
    spec= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
    sens = (cm1[1,1])/(cm1[1,0]+cm1[1,1])
    prec=cm1[1,1]/(cm1[1,1]+cm1[0,1])
    rec=cm1[1,1]/(cm1[1,1]+cm1[1,0])
    sensitivity.append(sens)
    specificity.append(spec)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    Nb_fpr_list.append(fpr)
    Nb_tpr_list.append(tpr)
    if max_accuracy<acc:
        max_accuracy=acc
        Nb_fpr=fpr
        Nb_tpr=tpr
        

Nb_fpr=pd.DataFrame(Nb_fpr)
Nb_tpr=pd.DataFrame(Nb_tpr)

Nb_fpr.to_csv('fpr_Nb.csv',header=False, index=False)
Nb_tpr.to_csv('tpr_Nb.csv',header=False, index=False)   
 
mean_acc=np.mean(accuracy)
std_acc=np.std(accuracy)

print("Accuracy:"+str(mean_acc)+" Â± "+str(std_acc))

mean_spec=np.mean(specificity)
std_spec=np.std(specificity)
print("Specificity:"+str(mean_spec)+" Â± "+str(std_spec))
mean_sens=np.mean(sensitivity)
std_sens=np.std(sensitivity)
print("Sensitivity:"+str(mean_sens)+" Â± "+str(std_sens))
mean_prec=np.mean(precision)
std_prec=np.std(precision)
print("Precison:"+str(mean_prec)+" Â± "+str(std_prec))
mean_rec=np.mean(recall)
std_rec=np.std(recall)
print("Recall:"+str(mean_rec)+" Â± "+str(std_rec))
mean_coef=np.mean(m_coef)
std_coef=np.std(m_coef)
print("MCC:"+str(mean_coef)+" Â± "+str(std_coef))

print("AUC:"+str(np.mean(auc_list)))
