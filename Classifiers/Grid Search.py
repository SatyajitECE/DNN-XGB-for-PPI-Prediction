
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, Dropout
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.models import Model
from keras.regularizers import l2,l1
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from xgboost import XGBClassifier
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def load_dataset():
    global df_pos,df_neg,X,y
    df_pos= pd.read_csv('PositiveYH.csv',header=None)
    df_neg = pd.read_csv('NegativeYH.csv',header=None)
    #df_neg=df_neg.sample(n=min(len(df_pos),len(df_neg)))
    #df_pos=df_pos.sample(n=min(len(df_pos),len(df_neg)))
    df_neg['Status'] = 0
    df_pos['Status'] = 1
    df_neg=df_neg.sample(n=len(df_pos))
    
    df = pd.concat([df_pos,df_neg])
    df = df.reset_index()
    df=df.sample(frac=1)
    df = df.iloc[:,1:]
    
    X = df.iloc[:,0:1986].values
    y = df.iloc[:,1986:].values
    
    scaler = RobustScaler().fit(X)
    X = scaler.transform(X)
    
    global X1_train,X2_train
    X1_train = X[:, :993]
    X2_train = X[:, 993:]


load_dataset()


def define_model():
     ########################################################"Channel-1" ########################################################
    
    input_1 = Input(shape=(993, ), name='Protein_a')
    p11 = Dense(512, activation='relu', kernel_initializer=W_init, name='ProA_feature_1', kernel_regularizer=W_reg)(input_1)
    p11 = Dropout(D_rate)(p11)
    
    p12 = Dense(256, activation='relu', kernel_initializer=W_init, name='ProA_feature_2', kernel_regularizer=W_reg)(p11)
    p12 = Dropout(D_rate)(p12)
    
    p13= Dense(128, activation='relu', kernel_initializer=W_init, name='ProA_feature_3', kernel_regularizer=W_reg)(p12)
    p13 = Dropout(D_rate)(p13)
    
    p14= Dense(64, activation='relu', kernel_initializer=W_init, name='ProA_feature_4', kernel_regularizer=W_reg)(p13)
    p14 = Dropout(D_rate)(p14)
    
    
    ########################################################"Channel-2" ########################################################
    
    input_2 = Input(shape=(993, ), name='Protein_b')
    p21 = Dense(512, activation='relu', kernel_initializer=W_init, name='ProB_feature_1', kernel_regularizer=W_reg)(input_2)
    p21 = Dropout(D_rate)(p21)
    
    p22 = Dense(256, activation='relu', kernel_initializer=W_init, name='ProB_feature_2', kernel_regularizer=W_reg)(p21)
    p22 = Dropout(D_rate)(p22)
    
    p23= Dense(128, activation='relu', kernel_initializer=W_init, name='ProB_feature_3', kernel_regularizer=W_reg)(p22)
    p23 = Dropout(D_rate)(p23)
    
    p24= Dense(64, activation='relu', kernel_initializer=W_init, name='ProB_feature_4', kernel_regularizer=W_reg)(p23)
    p24 = Dropout(D_rate)(p24)

    ##################################### Merge Abstraction features ##################################################
    
    merged = concatenate([p14,p24], name='merged_protein1_2')
    
    ##################################### Prediction Module ##########################################################
    
    pre_output = Dense(64, activation='relu', kernel_initializer=W_init, name='Merged_feature_1')(merged)
    pre_output = Dense(32, activation='relu', kernel_initializer=W_init, name='Merged_feature_2')(pre_output)
    pre_output = Dense(16, activation='relu', kernel_initializer=W_init, name='Merged_feature_3')(pre_output)

    
    pre_output=Dropout(D_rate)(pre_output)

    output = Dense(1, activation='sigmoid', name='output')(pre_output)
    model = Model(input=[input_1, input_2], output=output)
   
    model.compile(loss='binary_crossentropy', optimizer=L_method, metrics=['accuracy'])
    return model


################################################ Grid Search Module ###########################################
def define_parameters():
    
    global params
    params={}
    params["Learning Rate"]=[ 1, 0.1, 0.01, 0.001]
    params["Batch size"]=[16, 32, 64, 128]
    params["Momentum Rate"]=[0.8, 0.9]
    params["Weight Initialization"]=["uniform", "normal", "glorot_normal"," glorot_uniform"]
    params["Weight Regularization"]=[l2(0.01),l1(0.01)]
    params["Adaptive learning rate method"]=[SGD(lr=0.01, momentum=0.9, decay=0.001), ]
    params["Dropout rate"]=[0.1, 0.2, 0.5]
    params["Epochs"]=[10, 20, 30, 40, 50, 100]
    
    return params

def grid_search():
    
    params=define_parameters()
    max_accuracy=float("-inf")
    global max_params
    max_params={}
    
    for lr in params.get("Learning Rate"):
        
        global LR
        LR=lr
        
        for bs in params.get("Batch size"):
            
            global B_size
            B_size=bs
            
            for mr in params.get("Momentum Rate"):
                global M_rate
                M_rate=mr
                
                for wi in params.get("Weight Initialization"):
                    global W_init
                    W_init=wi
                    
                    for wr in params.get("Weight Regularization"):
                        global W_reg
                        W_reg=wr
                        
                        for alrm in params.get("Adaptive learning rate method"):
                            global L_method
                            L_method=alrm
                            
                            for dr in params.get("Dropout rate"):
                                global D_rate
                                D_rate=dr
                                
                                for epochs in params.get("Epochs"):
                                    global Epochs
                                    Epochs=epochs
                        
                                    kf=StratifiedKFold(n_splits=3)
                                    acc_list=[]
                                    global model
                                    for train, test in kf.split(X,y):
                                        model=define_model()
                                        
                                        model.fit([X1_train[train],X2_train[train]],y[train],epochs=Epochs,batch_size=B_size,verbose=1)
                                        global y_test
                                        y_test=y[test]
                                        global y_score
                                        y_score = model.predict([X1_train[test],X2_train[test]])
                                        
                                        for i in range(0,len(y_score)):
                                            if(y_score[i]>0.5):
                                                y_score[i]=1
                                            else:
                                                y_score[i]=0
                                                
                                        cm1=confusion_matrix(y_test,y_score)
                                        acc1 = (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
                                        acc_list.append(acc1)
                                    
                                    avg_acc=sum(acc_list)/3
                                    
                                    if avg_acc>max_accuracy:
                                        max_accuracy=avg_acc
                                        max_params["Learning Rate"]=LR
                                        max_params["Batch size"]=B_size
                                        max_params["Momentum Rate"]=M_rate
                                        max_params["Weight Initialization"]=W_init
                                        max_params["Weight Regularization"]=W_reg
                                        max_params["Adaptive learning rate method"]=L_method
                                        max_params["Dropout rate"]=D_rate
                                        max_params["Epochs"]=Epochs
    
    print(max_params)
    print()
    print("The average accuracy obtained is ",end=str(max_accuracy))
    print()

grid_search()