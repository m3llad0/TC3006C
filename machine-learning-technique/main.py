import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error



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
    
    return customTree.predict(predictData.values)

def evaluate_metrics(true_labels, predicted_labels):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    return accuracy, precision, recall, f1, conf_matrix

def evaluate_mse(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)

    return mse

def main():
    data = pd.read_csv('bmi_train.csv')

    # Dividir el conjunto de datos en entrenamiento y prueba
    trainingData, testData = train_test_split(data, test_size=0.2, random_state=42)

    #Entrenamiento de modelos
        # Entrenamiento de modelo custom
    customModel = trainingModel(trainingData)

    customPredictions = predictionModel(testData, customModel)

    print("Custom decision tree predictions:", customPredictions)

    accuracy, custom_precision, custom_recall, custom_f1, confusion_matrix = evaluate_metrics(testData['Index'], customPredictions)
    custom_mse = evaluate_mse(testData['Index'], customPredictions)  # Use a relevant column
    
    print("Metrics for Custom Decision Tree:")
    print("Accuracy:", accuracy)
    print("Precision:", custom_precision)
    print("Recall:", custom_recall)
    print("F1-score:", custom_f1)
    print("Mean Squared Error:", custom_mse)
    print("Confusion Matrix", confusion_matrix)

if __name__ == "__main__":
    main()