# Modelo de Regresión Lineal

Este proyecto contiene una implementación básica de un modelo de regresión lineal en Python 3.10.12. La regresión lineal es un enfoque de aprendizaje automático utilizado para modelar la relación entre una variable dependiente y una o más variables independientes mediante una ecuación lineal.

## Ejecución

Para ejecutar el archivo `main.py` y utilizar el modelo de regresión lineal, sigue estos pasos:

1. Asegúrate de tener Python 3.10.12 instalado. Puedes verificar la versión de Python ejecutando el siguiente comando en tu terminal:

    ```python --version```


2. Clona este repositorio en tu máquina local o descárgalo como un archivo ZIP.

3. Navega a la ubicación del archivo `main.py` en la carpeta `machine-learning-framework` en tu terminal.

4. Instala las librerías necesarias:

    ```pip install -r requirements.txt```


5. Ejecuta el archivo usando el siguiente comando:

    ```python main.py```


Esto ejecutará el script y mostrará ejemplos de cómo se entrena el modelo de regresión lineal y se hacen predicciones.

## Clase `LinearRegressionModel`

La clase `LinearRegressionModel` representa el modelo de regresión lineal y tiene los siguientes métodos y atributos:

- `__init__()`: Inicializa el objeto `LinearRegressionModel`.
- `fit(X_train, y_train)`: Ajusta el modelo de regresión lineal a los datos de entrenamiento.
- `predict(X_test)`: Realiza predicciones utilizando el modelo entrenado en nuevos datos.
- `evaluate(X_test, y_test)`: Evalúa el rendimiento del modelo de regresión lineal en datos de prueba.

### Uso

Aquí hay un ejemplo básico de cómo usar el modelo de regresión lineal:

```python
# Crear una instancia del modelo de regresión lineal
model = LinearRegressionModel()

# Entrenar el modelo en el conjunto de entrenamiento
X_train = ...
y_train = ...
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
X_test = ...
predictions = model.predict(X_test)

# Evaluar el modelo en el conjunto de prueba
y_test = ...
evaluation_result = model.evaluate(X_test, y_test)
```
Puedes utilizar este modelo para realizar análisis de regresión en tus datos y evaluar su rendimiento.