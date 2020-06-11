# surprisingly-popular-algorithm-ensemble
Implementazione dell'algoritmo "surprisingly popular" nel contesto di metodologie di apprendimento ensemble

# Analisi della letteratura
La documentazione utilizzata per ottenere una chiara visione d'insieme per l'implementazione dell'algoritmo si trova nella cartella "pubblicazioni". In essa sono presenti documenti relativi ai primi studi della materia, gli sviluppi degli ultimi anni eseguiti in diversi contesti, e varie altre pubblicazioni relative ai fondamenti dell'ambito di riferimento.

In particolare:
- *"A solution to the single-question crowd wisdom problem"*, *"Finding truth even if the crowd is wrong"* e *"A Bayesian Truth Serum for Subjective Data"* sono tre pubblicazioni fondamentali di Prelec D. (inventore dell'algoritmo);
- *"Testing the Ability of the Surprisingly Popular Algorithm to Predict the 2017 NBA Playoffs"* e *"Testing the ability of the surprisingly popular method to predict NFL games"*  sono due pubblicazioni per l'impementazione dell'algoritmo sulla base di predizioni;
- *"Wisdom of the Crowd: Comparison of theCWM, Simple Average and Surprisingly Popular Answer Method"* è la tesi di uno studente dell'università di Rotterdam che fornisce una comparazione tra diversi algoritmi tra cui il surprisingly popular;
- *"VOX POPULI"* pubblicazione del 1907 a cura di Galton come introduzione alla **saggezza della folla**.

# Software utilizzato
Il linguaggio utilizzato per l'implementazione è Python alla versione 3.7.4 attraverso la piattaforma Anaconda.
La stesura di codice è stata eseguita attraverso il tool Jupyter Notebook fornito direttamente da Anaconda.

# Codice
I file di implementazione sono inseriti nella cartella "codice" in due formati:
- con estensione *.ipynb* per l'esecuzione tramite Jupyter Notebook (consigliata);
- con estensione *.py* per l'esecuzione tramite shell;

In testa ad ogni file viene presentata la pipeline che si è seguita per l'implementazione dell'algoritmo.
Il codice sviluppato segue gli step definiti nei commenti inseriti in ogni cella, cercando di rendere il più possibile pulita l'esecuzione e l'interpretazione.
I file sono 4, uno per ogni metodologia ensemble adottata nell'implementazione: RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier e GradientBoostingClassifier.
I file, di default presentano l'utilizzo del toy dataset *load_iris* fornito da Scikit-learn, ma possono essere facilmente modificati sostituendo nelle prime righe il dataset da importare.

# Test performance
Lo script fornisce una comparazione basata sull'accuratezza tra la classificazione tramite implementazione del *surprisingly popular* con una particolare metodologia ensemble, e la stessa implementazione senza metodologia ensemble.
A favore di questa prospettiva, l'algoritmo è stato testato utilizzando sia predittori "deboli" sia predittori "forti", mettendo in risalto il fatto che la saggezza della folla basata su discenti "deboli" produce comunque risultati ottimi.
I test sono stati eseguiti su 4 differenti toy dataset al fine di produrre risultati eterogenei. I dataset utilizzati sono tutti forniti da Scikit-learn e sono: *load_iris*, *load_digits*, *load_wine*, *load_breast_cancer*.
I dati sulla performance sono consultabili nel file all'interno della cartella "test".

# Ulteriori suggerimenti
Per affrontare Python, machine learning, e sviluppo ensemble partendo da zero, ho trovato molto utile il libro che mi è stato consigliato: *"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow"* scritto da Aurélien Géron e pubblicato da O'Reilly.
