# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

juegos=pd.read_csv('vgsales.csv')

#reemplazo los valores nan por 0
juegos=juegos.replace(np.nan,"0")

#
juegos['Platform']=juegos['Platform'].replace("2600","Atari")

#convertimos los valores no numericos para poder laburar
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

#creo nuevas columnas convertidas
juegos["plataforma"]=encoder.fit_transform(juegos.Platform.values)
juegos["publica"]=encoder.fit_transform(juegos.Publisher.values)

#defino inputs y output
X=juegos[["plataforma","publica","Global_Sales"]]
y=juegos["Genre"]

#entrenamos y testeamos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#importo el clasificador
from sklearn.neural_network import MLPClassifier

#defino cómo voy a predecir

## adam
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),
                  max_iter=500,alpha=0.0001,
                  solver='adam',random_state=21,tol=0.000000001)
'''
##basico
mlp=MLPClassifier(hidden_layer_sizes=(5,5,5,5),max_iter=5000)

##lbfgs
mlp=MLPClassifier(hidden_layer_sizes=(6,6,6,6),solver='lbfgs',max_iter=6000)
'''

#entrenamos el modelo
mlp.fit(X_train,y_train)

#la matriz de confusion
predictions= mlp.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))