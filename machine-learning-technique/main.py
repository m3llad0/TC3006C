import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error


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

def predictionModel(predictData: pd.DataFrame, customTree: DecisionTree)->list:
    encoder = LabelEncoder()

    # Codificar variables categoricas
    predictData['Gender'] = encoder.fit_transform(predictData['Gender'])
    return customTree.predict(predictData.drop('Index', axis=1).values)

def main():
    data = pd.read_csv('bmi_train.csv')

    # Dividir el conjunto de datos en entrenamiento y prueba
    trainingData, testData = train_test_split(data, test_size=0.2, random_state=42)

    #Entrenamiento de modelos
        # Entrenamiento de modelo custom
    customModel = trainingModel(trainingData)
    customPredictions = predictionModel(testData, customModel)

    print("Decision Tree predictions:", customPredictions)
    
    print("Metrics for Decision Tree:")
    print(classification_report(testData['Index'], customPredictions))
    print("Mean Squared Error:", mean_squared_error(testData['Index'], customPredictions))
    print("Confusion Matrix", "\n", confusion_matrix(testData['Index'], customPredictions))

if __name__ == "__main__":
    main()