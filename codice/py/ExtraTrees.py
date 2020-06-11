''' Pipeline:
Caricare Dataset
Creare N Dataset Estratti Random (con Replacement)
Per ogni dataset:
    Addestri un modello che ti dice la classe predetta
Per ogni dataset:
    Addestri un modello che dice la probabilità delle varie classi
Metto insieme i loro risultati quando mi viene chiesta una predizione nuova
'''

#Caricare Dataset
import numpy as np
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

#divido il dataset in training set e testing set, e il training set ulteriormente in trainingVeroEProprio e Validation
#le proporzioni rispetto al totale sono: 60% trainVP, 20% validation, 20% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_trainVP, X_val, y_trainVP, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)  

'''(observed proportion)

Generate {D1,...,Dn} sampling with replacement from D
for all i = 1..n do
    Train decision tree Ti on Di
end for
'''

from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=100, max_depth=None, random_state=0)
etc.fit(X_trainVP, y_trainVP)

#array di decision tree basati sul training set vero e proprio
T = np.empty((etc.n_estimators), dtype = object)
for i in range (0, etc.n_estimators):
    T[i] = etc[i]

'''
E = Bagging-Ensemble(T1,...,Tn)
^y = E.predict(X)
'''

#y_pred è l'observed proportion basata sul validation set
y_pred = etc.predict(X_val)

''' (expected proportion)

for all i = 1..n do
    Train decision tree Pi on (X; ^y)
end for
'''

etc1 = ExtraTreesClassifier(n_estimators=100, max_depth=None, random_state=0)
etc1.fit(X_val, y_pred)

#array di decision tree basati su y_pred
P = np.empty_like(T)
for i in range (0, etc1.n_estimators):
    P[i] = etc1[i]

def new_prediction(sample):
    for i in range (0, etc.n_estimators):
        c[i] = T[i].predict(sample)
        p[i] = P[i].predict(sample)
    
    for i in range(0, etc.n_classes_):
        countc[i] = Counter(c)[i]
        countp[i] = Counter(p)[i]

    differences = countc - countp
    result = countc + differences
    
    max = np.argmax(result) #indice del valore massimo in result
    occurrences = np.count_nonzero(result == result[max]) #quante volte compare il massimo in result

    #gestione pareggio (in caso di parità viene scelta l'alternativa con observed proportion massima)
    if occurrences > 1:
        result1 = np.empty_like(result) #array in cui vengono inseriti i valori di countc corrispondenti al massimo in result
        for i in range(0, len(result1)):
            if (result[i] != result[max]):
                result1[i] = 0
            else:
                result1[i] = countc[i] #in corrispondenza dei valori massimi, inserisco in result1 i corrispondenti valori osservati (countc)
        max = np.argmax(result1) #indice del valore massimo in result1

    return max

'''
c = [0; ...; 0] s.t. |c| = |Y|
p = [0; ...; 0] s.t. |p| = |Y|
for all i = 1::n do
    c = c + Ti.predict(x)
    p = p + Pi.predict(x)
end for
'''

#array che per ogni decision tree definisce la classe predetta
#c è la predizione sulla base dell'observed
#p è la predizione sulla base dei predicted

from collections import Counter

c = np.empty((etc.n_estimators), dtype = int)
p = np.empty_like(c)

countc = np.empty((etc.n_classes_), dtype = int)
countp = np.empty_like(countc)
differences = np.empty_like(countc)
result = np.empty_like(countc)

predictions = np.empty((len(X_test)), dtype = int)

#popolo l'array con le predizioni sul test set
for i in range (0, len(X_test)):
    predictions[i] = new_prediction([X_test[i]])
    
#calcolo l'accuratezza
from sklearn.metrics import accuracy_score
acc_ET_SPA = accuracy_score(y_test, predictions)

#implementazione tramite ExtraTreeClassifier senza l'implementazione del surprisingly popular algorithm
etc3 = ExtraTreesClassifier(n_estimators=100, max_depth=None, random_state=0)

#calcolo l'accuratezza tramite cross-validation su tutto il dataset
from sklearn.model_selection import cross_val_score
scores = cross_val_score(etc3, X, y, cv=5)
acc_ET_single = scores.mean()

print("Accuratezza ExtraTrees tramite SPA: {:.4%}".format(acc_ET_SPA))
print("Accuratezza ExtraTrees non-SPA: {:.4%}".format(acc_ET_single))