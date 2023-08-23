import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier



"""
Implementación de un árbol de decisión desde cero
"""

def trainingModel(trainingData: pd.DataFrame)->DecisionTree:
    # Codificar variables categoricas
    encoder = LabelEncoder()
    trainingData['Gender'] = encoder.fit_transform(trainingData['Gender'])

    # Dividir los datos en caracteristicas y etiquetas de entrenamiento
    X_train = trainingData.drop('Index', axis=1).values
    y_train = trainingData['Index'].values

    # Crear y ajustar el modelo
    tree = DecisionTree(maxDepth = 100)
    tree.fit(X_train, y_train)

    return tree

def trainingSklearnModel(trainingData: pd.DataFrame)->DecisionTreeClassifier:
    #Codificar variables categoricas
    encoder = LabelEncoder()
    trainingData['Gender'] = encoder.fit_transform(trainingData['Gender'])

    # Dividir los datos en caracterisiticas y etiquetas de entrenamiento
    X_train = trainingData.drop('Index', axis=1).values
    y_train = trainingData['Index'].values

    tree = DecisionTreeClassifier(max_depth=100)
    tree.fit(X_train, y_train)

    return tree

def predictionModel(predictData: pd.DataFrame, customTree: DecisionTree)->list:
    encoder = LabelEncoder()

    # Codificar variables categoricas
    predictData['Gender'] = encoder.fit_transform(predictData['Gender'])
    
    return customTree.predict(predictData.values)


def predictonSklearn(predictData: pd.DataFrame, sklearnTree: DecisionTreeClassifier)->list:
    encoder = LabelEncoder()

    #Codficar variables categoricas
    predictData['Gender'] = encoder.fit_transform(predictData['Gender'])

    return sklearnTree.predict(predictData.values)


def main():
    # Carga de datos
    trainingData = pd.read_csv('bmi_train.csv')
    predictData = pd.read_csv('bmi_validation.csv')

    #Entrenamiento de modelos
        # Entrenamiento de modelo custom
    customModel = trainingModel(trainingData)
        # Entrenamiento de modelo sklearn
    sklearnModel = trainingSklearnModel(trainingData)

    #Generar predicciones
        #Entrenamiento de modelo custom
    customPredictions = predictionModel(predictData, customModel)
        #Entrenamiento de modelo de sklearn
    sklearnPredictions = predictonSklearn(predictData, sklearnModel)


    print("Custom decision tree predictions: ", customPredictions)
    print("Sklearn decision tree predictions", sklearnPredictions)




if __name__ == "__main__":
    main()