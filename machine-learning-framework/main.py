from linear_regression import LinearRegressionModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    data = pd.read_csv("bmi.csv")

    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])

    # Separar las características (X) y la variable objetivo (y)
    X = data.drop(columns=['Index'])
    y = data['Index']

    # Dividir los datos en conjuntos de entrenamiento y prueba (por ejemplo, 80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear una instancia del modelo
    model = LinearRegressionModel()

    # Entrenar el modelo en el conjunto de entrenamiento
    model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    print("Predicciones: ", y_pred)
    # Evaluar el modelo en el conjunto de prueba
    evaluation_result = model.evaluate(X_test, y_test)

    # Mostrar las métricas de evaluación
    print("Métricas de evaluación:")
    print("Mean Squared Error (MSE):", evaluation_result['Mean Squared Error'])
    print("R-squared (R^2):", evaluation_result['R-squared (R^2)'])
    print("Mean Absolute Error (MAE):", evaluation_result['Mean Absolute Error (MAE)'])

    # Crear una gráfica de regresion lineal
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Datos')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Línea de Regresión')
    plt.title('Gráfica de Regresión Lineal con Línea de Regresión')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()