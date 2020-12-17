import pandas as pd
import matplotlib.pyplot as plt

fprKNN=pd.read_csv("fprKNN.csv",header=None)
tprKNN=pd.read_csv("tprKNN.csv",header=None)
rocknn=0.815

fprNB=pd.read_csv("fprNB.csv",header=None)
tprNB=pd.read_csv("tprNB.csv",header=None)
rocnb=0.802

fprRF=pd.read_csv("fprRF1.csv",header=None)
tprRF=pd.read_csv("tprRF1.csv",header=None)
rocrf=0.990

fprAB=pd.read_csv("fprAb1.csv",header=None)
tprAB=pd.read_csv("tprAb1.csv",header=None)
rocab=0.939

fprXgb=pd.read_csv("fprXgb1.csv",header=None)
tprXgb=pd.read_csv("tprXgb1.csv",header=None)
rocXgb=0.971

fprSVM=pd.read_csv("fprSVM1.csv",header=None)
tprSVM=pd.read_csv("tprSVM1.csv",header=None)
rocsvm=0.929

fprdnn=pd.read_csv("fprdnn.csv",header=None)
tprdnn=pd.read_csv("tprdnn.csv",header=None)
rocdnn=0.950

fprdnnxgb=pd.read_csv("fprdnn_xgb.csv",header=None)
tprdnnxgb=pd.read_csv("tprdnn_xgb.csv",header=None)
rocdnnxgb=0.993

plt.figure(figsize=(8,6),dpi=300)

plt.plot(fprNB, tprNB, label='NB_ROC (area = %0.3f)' % rocnb)
plt.plot(fprKNN, tprKNN, label='KNN_ROC (area = %0.3f)' % rocknn)
plt.plot(fprSVM, tprSVM, label='SVM_ROC (area = %0.3f)' % rocsvm)
plt.plot(fprRF, tprRF, label='RF_ROC (area = %0.3f)' % rocrf)
plt.plot(fprAB, tprAB, label='Adaboost_ROC (area = %0.3f)' % rocab)
plt.plot(fprXgb, tprXgb, label='Xgboost_ROC (area = %0.3f)' % rocXgb)

plt.plot(fprdnn, tprdnn, label='DNN_ROC (area = %0.3f)' % rocdnn)
plt.plot(fprdnnxgb, tprdnnxgb, label='DNN-XGB_ROC (area = %0.3f)' % rocdnnxgb)

plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.tick_params(labelsize=15)
#plt.title('ROC curve comparison of various algorithms on S. cerevisiae data set')
plt.legend(loc='best')
plt.show()
