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
E = Bagging-Ensemble(T1,...,Tn)
'''

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

#array di AdaBoostClassifier basati sul training set vero e proprio
#ogni classificatore si basa su subset random del dataset originale
abc = AdaBoostClassifier(n_estimators=50, random_state=0)
T = BaggingClassifier(base_estimator=abc, n_estimators=100, bootstrap=True, random_state=0).fit(X_trainVP, y_trainVP)

'''
^y = E.predict(X)
'''

#y_pred è l'observed proportion basata sul validation set
y_pred = T.predict(X_val)

''' (expected proportion)

for all i = 1..n do
    Train decision tree Pi on (X; ^y)
end for
'''

#array di AdaBoostClassifier basati su y_pred
abc1 = AdaBoostClassifier(n_estimators=50, random_state=0)
P = BaggingClassifier(base_estimator=abc1, n_estimators=100, bootstrap=True, random_state=0).fit(X_val, y_pred)

def new_prediction(sample):
    for i in range (0, T.n_estimators):
        c[i] = T[i].predict(sample)
        p[i] = P[i].predict(sample)
    
    for i in range(0, T.n_classes_):
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

#array che per ogni AdaBoostClassifier definisce la classe predetta
#c è la predizione sulla base dell'observed
#p è la predizione sulla base dei predicted

from collections import Counter

c = np.empty((T.n_estimators), dtype = int)
p = np.empty_like(c)

countc = np.empty((T.n_classes_), dtype = int)
countp = np.empty_like(countc)
differences = np.empty_like(countc)
result = np.empty_like(countc)

predictions = np.empty((len(X_test)), dtype = int)

#popolo l'array con le predizioni sul test set
for i in range (0, len(X_test)):
    predictions[i] = new_prediction([X_test[i]])

#calcolo l'accuratezza
from sklearn.metrics import accuracy_score
acc_AB_SPA = accuracy_score(y_test, predictions)


#implementazione tramite AdaBoostClassifier senza l'implementazione del surprisingly popular algorithm
abc2 = AdaBoostClassifier(n_estimators=50, random_state=0)
bclf = BaggingClassifier(base_estimator=abc2, n_estimators=100, bootstrap=True, random_state=0)

#calcolo l'accuratezza tramite cross-validation su tutto il dataset
from sklearn.model_selection import cross_val_score
scores = cross_val_score(bclf, X, y, cv=5)
acc_AB_single = scores.mean()


print("Accuratezza AdaBoost tramite SPA: {:.4%}".format(acc_AB_SPA))
print("Accuratezza AdaBoost non-SPA: {:.4%}".format(acc_AB_single))