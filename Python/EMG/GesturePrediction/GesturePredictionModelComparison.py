# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 20:29:42 2021

@author: Ben
"""

#Final Project

##1 Data Restructuring
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


rock = pd.read_csv (r'C:\Users\Ben\OneDrive - purdue.edu\BME 511\FinalProject\gesture0.csv', header=None)
scissors = pd.read_csv (r'C:\Users\Ben\OneDrive - purdue.edu\BME 511\FinalProject\gesture1.csv', header=None)
paper = pd.read_csv (r'C:\Users\Ben\OneDrive - purdue.edu\BME 511\FinalProject\gesture2.csv', header=None)
ok = pd.read_csv (r'C:\Users\Ben\OneDrive - purdue.edu\BME 511\FinalProject\gesture3.csv', header=None)


from sklearn.model_selection import train_test_split#,cross_val_score,GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import classification_report,accuracy_score, f1_score

#from sklearn.svm import SVC 
#SVC broke my computer with larger datasets and thus was omitted
from sklearn.decomposition import PCA


fs = 200
t = np.arange(0,len(rock[0][:])/fs,1/fs) #converts the data into time domain for graphing    

s1 = [0,8,16,24,32,40,48,56]
sen = []

for i in range(0,8):
    sen.append([x+i for x in s1]) #sets organization list for the sensor data to attach to.



def sense_seperate(inputdata,sen): #seperates each sensor and each data
    sensordata = []
    for i in range(0,8):
        sensordata.append(inputdata[inputdata.columns[sen[i]]])
    
    return(sensordata)


sensorrock = sense_seperate(rock,sen)
sensorscissors = sense_seperate(scissors,sen)
sensorpaper = sense_seperate(paper,sen)
sensorok = sense_seperate(ok,sen)
#sensorall = sense_seperate(allData,sen)


def flatten_list (_3d_list): #flattens out trials

    newarr = np.asarray(_3d_list) #makes it an array which should allow me to concatenate??
    _2darr = newarr.reshape(newarr.shape[0], (newarr.shape[1]*newarr.shape[2]))
    time = np.arange(0,len(_2darr[0][:])/fs,1/fs)
    
    return(_2darr,time)


def SensorPlot(dataset, time, name): #plots each sensor 
    plt.figure()
    
    for i in range(0,8):
        plt.subplot(2,4,i+1, )#sharex=True, sharey=True)
        plt.plot(time,dataset[i])
        plt.title('Sensor'+str(i+1))
    plt.suptitle(str(name))
    plt.tight_layout()
    
    return()


sensorrockFlat,trock = flatten_list(sensorrock)
sensorscissorsFlat,tscissors = flatten_list(sensorscissors)
sensorpaperFlat,tpaper = flatten_list(sensorpaper)
sensorokFlat,tok = flatten_list(sensorok)
#sensorallFlat,tall = flatten_list(sensorall)

claRock = [0] * len(trock)
claScissors = [1] *len(tscissors)
claPaper = [2] * len(tpaper)
claOk =  [3] * len(tok)

classifierVector = np.concatenate((claRock, claScissors,claPaper, claOk), axis = 0)

rockName = "Rock Sensors"
scissorsName = "Scissor Sensors"
paperName = "Paper Sensors"
okName = "Ok Sensors"
allName = "All Gestured Data per Sensor (Concatenated)"

SensorPlot(sensorrockFlat,trock,rockName)
SensorPlot(sensorscissorsFlat,tscissors,scissorsName)
SensorPlot(sensorpaperFlat,tpaper,paperName)
SensorPlot(sensorokFlat,tok,okName)





##2. Key Characteristic Selection and Time Windowing

from sklearn.preprocessing import StandardScaler #not sure why this is yellow, it is used

def MeanReduction(inputdata,times): #Key characteristic to define and window data.
    inputdata = abs(inputdata)
    OneSecData = np.empty((8,len(np.transpose(inputdata))))
    
    times = int(times)
    for i in range(0,len(inputdata)): #iterates per senso
        for j in range(0,len(np.transpose(inputdata))):   #long ways
           
            if (j % times) == 0 :
                OneSecData[i][j] = np.mean(inputdata[i][j:j+times]) #Error is currently here and not sure what it is
                
    ReducedData = []
    for i in range(0,len(OneSecData)):  
        tempdata = []
        for j in range(0,len(OneSecData[0])):
            if OneSecData[i][j] != 0:
                tempdata.append(OneSecData[i][j])
        ReducedData.append(tempdata)
        
    #ReducedData = np.transpose(ReducedData)
    ReducedData = np.asarray(ReducedData)
    return(ReducedData)


onesec = fs #fs = 1 second,  if you want half sec fs/2
halfsec = fs/2

ReducedRock = MeanReduction(sensorrockFlat,onesec)
ReducedScissors = MeanReduction(sensorscissorsFlat,onesec)
ReducedPaper = MeanReduction(sensorpaperFlat,onesec)
ReducedOk = MeanReduction(sensorokFlat,onesec)


ReducedAll =np.concatenate((ReducedRock,ReducedScissors,ReducedPaper, ReducedOk),axis = 1) #+ ReducedPaper + ReducedOk
tpReducedAll = np.transpose(ReducedAll)





##3. Model Application (sorry for hardcoding, the project progressed differently than expected :( 
##if worked on infuture all of this would be a single function and called 25 times)

clasRock = [0] * len(ReducedRock[1])
clasScissors = [1] *len(ReducedScissors[1])
clasPaper = [2] * len(ReducedPaper[1])
clasOk =  [3] * len(ReducedOk[1])

clasVector = np.concatenate((clasRock, clasScissors,clasPaper, clasOk), axis = 0)

X_train1,X_test1,Y_train1,Y_test1=train_test_split(tpReducedAll,clasVector,train_size = .70, test_size=0.3,random_state=0,stratify = clasVector)

#MLP Model
scaler=StandardScaler()
scaler.fit(X_train1)
X_train_scaled=scaler.transform(X_train1)
X_test_scaled=scaler.transform(X_test1)

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier().fit(X_train_scaled,Y_train1)
y_pred=mlp.predict(X_test_scaled)
y_predtrain = mlp.predict(X_train_scaled)

AccMLPTest = accuracy_score(Y_test1,y_pred) * 100
AccMLPTrain = accuracy_score(Y_train1,y_predtrain) * 100

print(f'MLP Training accuracy for 1 second of Data = {AccMLPTest:0.1f}%, Test accuracy = {AccMLPTrain:0.1f}%')
print(classification_report(Y_test1,y_pred))

#Linear Model
C = 1 #hyperparameter
modelL = linear_model.LogisticRegression(penalty='l2', solver='liblinear', C=C)
modelL.fit(X_train1,Y_train1) #EMG data tends to be Guassian in nature with a mean of zero

Y_predicted_train1 = modelL.predict(X_train1)
Y_predicted_test1 = modelL.predict(X_test1)

acc_train = (Y_predicted_train1 == Y_train1).sum() * 100. / Y_train1.shape[0]
acc_test = (Y_predicted_test1 == Y_test1).sum() * 100. / Y_test1.shape[0]

print(f'Linear Training accuracy for 1 Second of Data = {acc_train:0.1f}%, Test accuracy = {acc_test:0.1f}%')

#Gaussian Model
modelG=GaussianNB()
modelG.fit(X_train1,Y_train1) #EMG data tends to be Guassian in nature with a mean of zero

Y_predicted_train1 = modelG.predict(X_train1)
Y_predicted_test1 = modelG.predict(X_test1)

acc_train = (Y_predicted_train1 == Y_train1).sum() * 100. / Y_train1.shape[0]
acc_test = (Y_predicted_test1 == Y_test1).sum() * 100. / Y_test1.shape[0]

print(f'NBGaussian Training accuracy for 1 second of Data = {acc_train:0.1f}%, Test accuracy = {acc_test:0.1f}%')

#KNN Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

params={"n_neighbors": np.arange(1,10)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,params,cv=10)
knn_cv.fit(X_train1,Y_train1)

bestparams = knn_cv.best_params_
knn_model=KNeighborsClassifier(n_neighbors=8)
knn_tuned=knn_model.fit(X_train1,Y_train1)

y_pred=knn_tuned.predict(X_test1)
y_predtrain=knn_model.predict(X_train1)

AccKnnTest = accuracy_score(Y_test1,y_pred)* 100
AccKnnTrain = accuracy_score(Y_train1,y_predtrain)* 100

print(f'Optimized KNN Training accuracy for 1 second of Data = {AccKnnTest:0.1f}%, Test accuracy = {AccKnnTrain:0.1f}%')

#PCA Model
pc = PCA(n_components=8)
pc.fit(X_train1)

X_train_pc = pc.transform(X_train1)
X_test_pc = pc.transform(X_test1)

modelP = GaussianNB()
modelP.fit(X_train_pc, Y_train1)

y_train_predicted = modelP.predict(X_train_pc)
y_predicted_test1 = modelP.predict(X_test_pc)

acc_train2 = 100 * (y_train_predicted == Y_train1).sum() / Y_train1.shape[0]
acc_test = (y_predicted_test1 == Y_test1).sum() * 100. / Y_test1.shape[0]
print(f'PCA NBGaussian Training accuracy for 1 second of Data = {acc_train2:0.1f}%, Test accuracy = {acc_test:0.1f}%')



print('')
print('Using only 0.5 Seconds of Data')
HalfsecRock = MeanReduction(sensorrockFlat,halfsec)
HalfsecScissors = MeanReduction(sensorscissorsFlat,halfsec)
HalfsecPaper = MeanReduction(sensorpaperFlat,halfsec)
HalfsecOk = MeanReduction(sensorokFlat,halfsec)

HalfsecAll =np.concatenate((HalfsecRock,HalfsecScissors,HalfsecPaper, HalfsecOk),axis = 1) #+ ReducedPaper + ReducedOk
tpHalfsecAll = np.transpose(HalfsecAll)

hclasRock = [0] * len(HalfsecRock[1])
hclasScissors = [1] *len(HalfsecScissors[1])
hclasPaper = [2] * len(HalfsecPaper[1])
hclasOk =  [3] * len(HalfsecOk[1])

hclasVector = np.concatenate((hclasRock, hclasScissors,hclasPaper,hclasOk), axis = 0)

X_train1,X_test1,Y_train1,Y_test1=train_test_split(tpHalfsecAll,hclasVector,train_size = .70, test_size=0.3,random_state=0,stratify = hclasVector)

#MLP Model
scaler=StandardScaler()
scaler.fit(X_train1)
X_train_scaled=scaler.transform(X_train1)
X_test_scaled=scaler.transform(X_test1)

mlp=MLPClassifier().fit(X_train_scaled,Y_train1)
y_pred=mlp.predict(X_test_scaled)
y_predtrain = mlp.predict(X_train_scaled)

AccMLPTest = accuracy_score(Y_test1,y_pred) * 100
AccMLPTrain = accuracy_score(Y_train1,y_predtrain) * 100

print(f'MLP Training accuracy for 1/2 second of Data = {AccMLPTest:0.1f}%, Test accuracy = {AccMLPTrain:0.1f}%')
print(classification_report(Y_test1,y_pred))

#Linear Model
C = 1 #hyperparameter
modelL = linear_model.LogisticRegression(penalty='l2', solver='liblinear', C=C)
modelL.fit(X_train1,Y_train1) #EMG data tends to be Guassian in nature with a mean of zero

Y_predicted_train1 = modelL.predict(X_train1)
Y_predicted_test1 = modelL.predict(X_test1)

acc_train = (Y_predicted_train1 == Y_train1).sum() * 100. / Y_train1.shape[0]
acc_test = (Y_predicted_test1 == Y_test1).sum() * 100. / Y_test1.shape[0]

print(f'Linear Training accuracy for 1/2 Second of Data = {acc_train:0.1f}%, Test accuracy = {acc_test:0.1f}%')

#Gaussian Model
modelG=GaussianNB()
modelG.fit(X_train1,Y_train1) #EMG data tends to be Guassian in nature with a mean of zero

Y_predicted_train1 = modelG.predict(X_train1)
Y_predicted_test1 = modelG.predict(X_test1)

acc_train = (Y_predicted_train1 == Y_train1).sum() * 100. / Y_train1.shape[0]
acc_test = (Y_predicted_test1 == Y_test1).sum() * 100. / Y_test1.shape[0]

print(f'NBGaussian Training accuracy for 1/2 second of Data = {acc_train:0.1f}%, Test accuracy = {acc_test:0.1f}%')

#KNN Model

params={"n_neighbors": np.arange(1,10)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,params,cv=10)
knn_cv.fit(X_train1,Y_train1)

bestparams = knn_cv.best_params_
knn_model=KNeighborsClassifier(n_neighbors=8)
knn_tuned=knn_model.fit(X_train1,Y_train1)

y_pred=knn_tuned.predict(X_test1)
y_predtrain=knn_model.predict(X_train1)

AccKnnTest = accuracy_score(Y_test1,y_pred)* 100
AccKnnTrain = accuracy_score(Y_train1,y_predtrain)* 100

print(f'Optimized KNN Training accuracy for 1/2 second of Data = {AccKnnTest:0.1f}%, Test accuracy = {AccKnnTrain:0.1f}%')

#PCA Model
pc = PCA(n_components=8)
pc.fit(X_train1)

X_train_pc = pc.transform(X_train1)
X_test_pc = pc.transform(X_test1)

modelP = GaussianNB()
modelP.fit(X_train_pc, Y_train1)

y_train_predicted = modelP.predict(X_train_pc)
y_predicted_test1 = modelP.predict(X_test_pc)

acc_train2 = 100 * (y_train_predicted == Y_train1).sum() / Y_train1.shape[0]
acc_test = (y_predicted_test1 == Y_test1).sum() * 100. / Y_test1.shape[0]
print(f'PCA NBGaussian Training accuracy for 1/2 second of Data = {acc_train2:0.1f}%, Test accuracy = {acc_test:0.1f}%')


print('')
print('Using only 0.25 Seconds of Data')

qsec = fs / 4

QuartersecRock = MeanReduction(sensorrockFlat,qsec)
QuartersecScissors = MeanReduction(sensorscissorsFlat,qsec)
QuartersecPaper = MeanReduction(sensorpaperFlat,qsec)
QuartersecOk = MeanReduction(sensorokFlat,qsec)

QuartersecAll =np.concatenate((QuartersecRock,QuartersecScissors,QuartersecPaper, QuartersecOk),axis = 1) #+ ReducedPaper + ReducedOk
tpQuartersecAll = np.transpose(QuartersecAll)

qclasRock = [0] * len(QuartersecRock[1])
qclasScissors = [1] *len(QuartersecScissors[1])
qclasPaper = [2] * len(QuartersecPaper[1])
qclasOk =  [3] * len(QuartersecOk[1])

qclasVector = np.concatenate((qclasRock, qclasScissors,qclasPaper,qclasOk), axis = 0)

X_train2,X_test2,Y_train2,Y_test2=train_test_split(tpQuartersecAll,qclasVector,train_size = .70, test_size=0.3,random_state=0,stratify = qclasVector)

#MLP Model
scaler=StandardScaler()
scaler.fit(X_train2)
X_train_scaled=scaler.transform(X_train2)
X_test_scaled=scaler.transform(X_test2)

mlp=MLPClassifier().fit(X_train_scaled,Y_train2)
y_pred=mlp.predict(X_test_scaled)
y_predtrain = mlp.predict(X_train_scaled)

AccMLPTest = accuracy_score(Y_test2,y_pred) * 100
AccMLPTrain = accuracy_score(Y_train2,y_predtrain) * 100

print(f'MLP Training accuracy for 1/4 second of Data = {AccMLPTest:0.1f}%, Test accuracy = {AccMLPTrain:0.1f}%')
print(classification_report(Y_test2,y_pred))

#Linear Model
C = 1 #hyperparameter
modelL = linear_model.LogisticRegression(penalty='l2', solver='liblinear', C=C)
modelL.fit(X_train2,Y_train2) #EMG data tends to be Guassian in nature with a mean of zero

Y_predicted_train2 = modelL.predict(X_train2)
Y_predicted_test2 = modelL.predict(X_test2)

acc_train = (Y_predicted_train2 == Y_train2).sum() * 100. / Y_train2.shape[0]
acc_test = (Y_predicted_test2 == Y_test2).sum() * 100. / Y_test2.shape[0]

print(f'Linear Training accuracy for 1/4 Second of Data = {acc_train:0.1f}%, Test accuracy = {acc_test:0.1f}%')

#Gaussian Model
modelG=GaussianNB()
modelG.fit(X_train2,Y_train2) #EMG data tends to be Guassian in nature with a mean of zero

Y_predicted_train2 = modelG.predict(X_train2)
Y_predicted_test2 = modelG.predict(X_test2)

acc_train = (Y_predicted_train2 == Y_train2).sum() * 100. / Y_train2.shape[0]
acc_test = (Y_predicted_test2 == Y_test2).sum() * 100. / Y_test2.shape[0]

print(f'NBGaussian Training accuracy for 1/4 second of Data = {acc_train:0.1f}%, Test accuracy = {acc_test:0.1f}%')

#KNN Model

params={"n_neighbors": np.arange(1,10)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,params,cv=10)
knn_cv.fit(X_train2,Y_train2)

bestparams = knn_cv.best_params_
knn_model=KNeighborsClassifier(n_neighbors=8)
knn_tuned=knn_model.fit(X_train2,Y_train2)

y_pred=knn_tuned.predict(X_test2)
y_predtrain=knn_model.predict(X_train2)

AccKnnTest = accuracy_score(Y_test2,y_pred)* 100
AccKnnTrain = accuracy_score(Y_train2,y_predtrain)* 100

print(f'Optimized KNN Training accuracy for 1/4 second of Data = {AccKnnTest:0.1f}%, Test accuracy = {AccKnnTrain:0.1f}%')

#PCA Model
pc = PCA(n_components=8)
pc.fit(X_train2)

X_train_pc = pc.transform(X_train2)
X_test_pc = pc.transform(X_test2)

modelP = GaussianNB()
modelP.fit(X_train_pc, Y_train2)

y_train_predicted = modelP.predict(X_train_pc)
y_predicted_test2 = modelP.predict(X_test_pc)

acc_train2 = 100 * (y_train_predicted == Y_train2).sum() / Y_train2.shape[0]
acc_test = (y_predicted_test2 == Y_test2).sum() * 100. / Y_test2.shape[0]
print(f'PCA NBGaussian Training accuracy for 1/4 second of Data = {acc_train2:0.1f}%, Test accuracy = {acc_test:0.1f}%')


print('')
print('Using only 0.05 Seconds of Data')

Twentsec = fs / 20

TwentsecRock = MeanReduction(sensorrockFlat,Twentsec)
TwentsecScissors = MeanReduction(sensorscissorsFlat,Twentsec)
TwentsecPaper = MeanReduction(sensorpaperFlat,Twentsec)
TwentsecOk = MeanReduction(sensorokFlat,Twentsec)

TwentsecAll =np.concatenate((TwentsecRock,TwentsecScissors,TwentsecPaper, TwentsecOk),axis = 1) #+ ReducedPaper + ReducedOk
tpTwentsecAll = np.transpose(TwentsecAll)

TclasRock = [0] * len(TwentsecRock[1])
TclasScissors = [1] *len(TwentsecScissors[1])
TclasPaper = [2] * len(TwentsecPaper[1])
TclasOk =  [3] * len(TwentsecOk[1])

TclasVector = np.concatenate((TclasRock, TclasScissors,TclasPaper,TclasOk), axis = 0)

X_train3,X_test3,Y_train3,Y_test3=train_test_split(tpTwentsecAll,TclasVector,train_size = .70, test_size=0.3,random_state=0,stratify = TclasVector)

#MLP Model
scaler=StandardScaler()
scaler.fit(X_train3)
X_train_scaled=scaler.transform(X_train3)
X_test_scaled=scaler.transform(X_test3)

mlp=MLPClassifier().fit(X_train_scaled,Y_train3)
y_pred=mlp.predict(X_test_scaled)
y_predtrain = mlp.predict(X_train_scaled)

AccMLPTest = accuracy_score(Y_test3,y_pred) * 100
AccMLPTrain = accuracy_score(Y_train3,y_predtrain) * 100

print(f'MLP Training accuracy for 1/20th second of Data = {AccMLPTest:0.1f}%, Test accuracy = {AccMLPTrain:0.1f}%')
print(classification_report(Y_test3,y_pred))

#Linear Model
C = 1 #hyperparameter
modelL = linear_model.LogisticRegression(penalty='l2', solver='liblinear', C=C)
modelL.fit(X_train3,Y_train3) #EMG data tends to be Guassian in nature with a mean of zero

Y_predicted_train3 = modelL.predict(X_train3)
Y_predicted_test3 = modelL.predict(X_test3)

acc_train = (Y_predicted_train3 == Y_train3).sum() * 100. / Y_train3.shape[0]
acc_test = (Y_predicted_test3 == Y_test3).sum() * 100. / Y_test3.shape[0]

print(f'Linear Training accuracy for 1/20th Second of Data = {acc_train:0.1f}%, Test accuracy = {acc_test:0.1f}%')

#Gaussian Model
modelG=GaussianNB()
modelG.fit(X_train3,Y_train3) #EMG data tends to be Guassian in nature with a mean of zero

Y_predicted_train3 = modelG.predict(X_train3)
Y_predicted_test3 = modelG.predict(X_test3)

acc_train = (Y_predicted_train3 == Y_train3).sum() * 100. / Y_train3.shape[0]
acc_test = (Y_predicted_test3 == Y_test3).sum() * 100. / Y_test3.shape[0]

print(f'NBGaussian Training accuracy for 1/20th second of Data = {acc_train:0.1f}%, Test accuracy = {acc_test:0.1f}%')

#KNN Model

params={"n_neighbors": np.arange(1,10)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,params,cv=10)
knn_cv.fit(X_train3,Y_train3)

bestparams = knn_cv.best_params_
knn_model=KNeighborsClassifier(n_neighbors=8)
knn_tuned=knn_model.fit(X_train3,Y_train3)

y_pred=knn_tuned.predict(X_test3)
y_predtrain=knn_model.predict(X_train3)

AccKnnTest = accuracy_score(Y_test3,y_pred)* 100
AccKnnTrain = accuracy_score(Y_train3,y_predtrain)* 100

print(f'Optimized KNN Training accuracy for 1/20th second of Data = {AccKnnTest:0.1f}%, Test accuracy = {AccKnnTrain:0.1f}%')

#PCA Model
pc = PCA(n_components=8)
pc.fit(X_train3)

X_train_pc = pc.transform(X_train3)
X_test_pc = pc.transform(X_test3)

modelP = GaussianNB()
modelP.fit(X_train_pc, Y_train3)

y_train_predicted = modelP.predict(X_train_pc)
y_predicted_test3 = modelP.predict(X_test_pc)

acc_train2 = 100 * (y_train_predicted == Y_train3).sum() / Y_train3.shape[0]
acc_test = (y_predicted_test3 == Y_test3).sum() * 100. / Y_test3.shape[0]
print(f'PCA NBGaussian Training accuracy for 1/20th second of Data = {acc_train2:0.1f}%, Test accuracy = {acc_test:0.1f}%')


print('')
print('Using only 0.005 Seconds of Data')

Twentsec = fs / 20

TwentsecRock = MeanReduction(sensorrockFlat,Twentsec)
TwentsecScissors = MeanReduction(sensorscissorsFlat,Twentsec)
TwentsecPaper = MeanReduction(sensorpaperFlat,Twentsec)
TwentsecOk = MeanReduction(sensorokFlat,Twentsec)

TwentsecAll =np.concatenate((TwentsecRock,TwentsecScissors,TwentsecPaper, TwentsecOk),axis = 1) #+ ReducedPaper + ReducedOk
tpTwentsecAll = np.transpose(TwentsecAll)

TclasRock = [0] * len(TwentsecRock[1])
TclasScissors = [1] *len(TwentsecScissors[1])
TclasPaper = [2] * len(TwentsecPaper[1])
TclasOk =  [3] * len(TwentsecOk[1])

TclasVector = np.concatenate((TclasRock, TclasScissors,TclasPaper,TclasOk), axis = 0)

X_train3,X_test3,Y_train3,Y_test3=train_test_split(tpTwentsecAll,TclasVector,train_size = .70, test_size=0.3,random_state=0,stratify = TclasVector)

#MLP Model
scaler=StandardScaler()
scaler.fit(X_train3)
X_train_scaled=scaler.transform(X_train3)
X_test_scaled=scaler.transform(X_test3)

mlp=MLPClassifier().fit(X_train_scaled,Y_train3)
y_pred=mlp.predict(X_test_scaled)
y_predtrain = mlp.predict(X_train_scaled)

AccMLPTest = accuracy_score(Y_test3,y_pred) * 100
AccMLPTrain = accuracy_score(Y_train3,y_predtrain) * 100

print(f'MLP Training accuracy for 1/20th second of Data = {AccMLPTest:0.1f}%, Test accuracy = {AccMLPTrain:0.1f}%')
print(classification_report(Y_test3,y_pred))

from sklearn import metrics 

#Linear Model
C = 1 #hyperparameter
modelL = linear_model.LogisticRegression(penalty='l2', solver='liblinear', C=C)
modelL.fit(X_train3,Y_train3) #EMG data tends to be Guassian in nature with a mean of zero

Y_predicted_train3 = modelL.predict(X_train3)
Y_predicted_test3 = modelL.predict(X_test3)

acc_train = (Y_predicted_train3 == Y_train3).sum() * 100. / Y_train3.shape[0]
acc_test = (Y_predicted_test3 == Y_test3).sum() * 100. / Y_test3.shape[0]

print(f'Linear Training accuracy for 1/20th Second of Data = {acc_train:0.1f}%, Test accuracy = {acc_test:0.1f}%')


#KNN Model

params={"n_neighbors": np.arange(1,10)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,params,cv=10)
knn_cv.fit(X_train3,Y_train3)

bestparams = knn_cv.best_params_
knn_model=KNeighborsClassifier(n_neighbors=8)
knn_tuned=knn_model.fit(X_train3,Y_train3)

y_pred=knn_tuned.predict(X_test3)
y_predtrain=knn_model.predict(X_train3)

AccKnnTest = accuracy_score(Y_test3,y_pred)* 100
AccKnnTrain = accuracy_score(Y_train3,y_predtrain)* 100

print(f'Optimized KNN Training accuracy for 1/20th second of Data = {AccKnnTest:0.1f}%, Test accuracy = {AccKnnTrain:0.1f}%')

print('')
print('Using only 0.0005 Seconds of Data, a Single EMG Reading at 200Hz')

Fullsec = fs / 200

FullsecAll =np.concatenate((sensorrockFlat,sensorscissorsFlat,sensorpaperFlat,sensorokFlat),axis = 1) #+ ReducedPaper + ReducedOk
tpFullsecAll = np.transpose(FullsecAll)

FullclasRock = [0] * len(sensorrockFlat[1])
FullclasScissors = [1] *len(sensorscissorsFlat[1])
FullclasPaper = [2] * len(sensorpaperFlat[1])
FullclasOk =  [3] * len(sensorokFlat[1])

FclasVector = np.concatenate((FullclasRock, FullclasScissors,FullclasPaper,FullclasOk), axis = 0)

X_train4,X_test4,Y_train4,Y_test4=train_test_split(tpFullsecAll,FclasVector,train_size = .70, test_size=0.3,random_state=0,stratify = FclasVector)

#MLP Model
scaler=StandardScaler()
scaler.fit(X_train4)
X_train_scaled=scaler.transform(X_train4)
X_test_scaled=scaler.transform(X_test4)

mlp=MLPClassifier().fit(X_train_scaled,Y_train4)
y_pred=mlp.predict(X_test_scaled)
y_predtrain = mlp.predict(X_train_scaled)

AccMLPTest = accuracy_score(Y_test4,y_pred) * 100
AccMLPTrain = accuracy_score(Y_train4,y_predtrain) * 100

print(f'MLP Training accuracy for a Single Data reading = {AccMLPTest:0.1f}%, Test accuracy = {AccMLPTrain:0.1f}%')
print(classification_report(Y_test4,y_pred))

#Linear Model
C = 1 #hyperparameter
modelL = linear_model.LogisticRegression(penalty='l2', solver='liblinear', C=C)
modelL.fit(X_train3,Y_train3) #EMG data tends to be Guassian in nature with a mean of zero

Y_predicted_train4 = modelL.predict(X_train4)
Y_predicted_test4 = modelL.predict(X_test4)

acc_train = (Y_predicted_train4 == Y_train4).sum() * 100. / Y_train4.shape[0]
acc_test = (Y_predicted_test4 == Y_test4).sum() * 100. / Y_test4.shape[0]

print(f'Linear Training accuracy for a Single Data reading= {acc_train:0.1f}%, Test accuracy = {acc_test:0.1f}%')


#KNN Model
params={"n_neighbors": np.arange(1,10)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,params,cv=10)
knn_cv.fit(X_train4,Y_train4)

bestparams = knn_cv.best_params_
knn_model=KNeighborsClassifier(n_neighbors=8)
knn_tuned=knn_model.fit(X_train4,Y_train4)

y_pred=knn_tuned.predict(X_test4)
y_predtrain=knn_model.predict(X_train4)

AccKnnTest = accuracy_score(Y_test4,y_pred)* 100
AccKnnTrain = accuracy_score(Y_train4,y_predtrain)* 100

print(f'Optimized KNN Training accuracy for a Single Data reading = {AccKnnTest:0.1f}%, Test accuracy = {AccKnnTrain:0.1f}%')





## 4. Confusion Matrix and other diagnostic plots

from sklearn import metrics 
confusion = metrics.confusion_matrix(Y_test4, y_pred) 

Gestures = ['Rock','Scissors','Paper','OK']
confusionplot1 = metrics.plot_confusion_matrix(mlp,X_test4,Y_test4, normalize = 'true', display_labels = Gestures)
confusionplot2 = metrics.plot_confusion_matrix(mlp,X_test4,Y_test4, normalize = 'pred')
confusionplot3 = metrics.plot_confusion_matrix(mlp,X_test4,Y_test4, normalize = 'all')


def getThresh(dataset):
    
    
    vardata = np.var(dataset,axis=1) #simple variance of each sensor
    avgdata = np.mean(abs(dataset),axis = 1) #takes absolute value of all data and takes the mean
    VarofVar = np.var(vardata) #variance between each EMG sensor, higher means some are activated mroe than others?
    
    tpdataset = np.transpose(dataset)
    SummedVar = sum(abs(tpdataset))#total variance of all 8 sensors - higher means more activation of Electrodes
    return(vardata,avgdata,VarofVar,SummedVar)

varAllsens, avgSens, VoVarAll, SumVar = getThresh(X_train1)

#Extra Plots

plt.figure()
plt.scatter(Y_train1,varAllsens)
plt.title('Varience of all 8 Sensors per Gesture')
plt.xlabel('Rock                       Scissors                       Paper                          OK ')
plt.ylabel('Total Variance')


plt.figure()
plt.scatter(Y_train1,avgSens)
plt.title('Average Absolute Value of each Sensor per Gesture')
plt.xlabel('Rock                       Scissors                       Paper                          OK ')
plt.ylabel('Average Absolute Value')

plt.figure()
plt.scatter(Y_train1,SumVar)
plt.title('Summed Absolute Value of each Sensor per Gesture')
plt.xlabel('Rock                       Scissors                       Paper                          OK ')
plt.ylabel('Summed Absolute Value')


varRock,avgRock,VoVarRock,sumRockV = getThresh(sensorrockFlat)
varScissors,avgScissors,VoVarScissors,sumScissorsV   = getThresh(sensorscissorsFlat)
varPaper,avgPaper,VoVarPaper,sumPaperV   = getThresh(sensorpaperFlat)
varOk,avgOk,VoVarOk,sumOkV   = getThresh(sensorokFlat)
plt.figure()
for i in range (0,8):
    plt.scatter(i+1,varRock[i], c = "red")
    plt.scatter(i+1,varScissors[i], c = "blue")
    plt.scatter(i+1,varPaper[i], c = "green")
    plt.scatter(i+1,varOk[i], c = "yellow")
plt.xlabel('Sensor')
plt.ylabel('Variance')
plt.title('Total Variance per Sensor')

