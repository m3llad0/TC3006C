# Decision Tree Classifier

Este proyecto contiene una implementación básica de un clasificador de árbol de decisiones en Python 3.10.12. Un árbol de decisiones es una estructura de modelo de aprendizaje automático que divide los datos en varias ramas basadas en las características de entrada, permitiendo la clasificación de nuevas muestras.

## Clase `Node`

La clase `Node` representa un nodo en un árbol de decisiones y tiene los siguientes atributos:

- `featureIndex`: El índice de la característica utilizada para dividir en este nodo.
- `threshold`: El valor de umbral utilizado para dividir en este nodo.
- `left`: El nodo hijo izquierdo.
- `right`: El nodo hijo derecho.
- `value`: El valor de la etiqueta si este nodo es una hoja.

## Clase `DecisionTree`

La clase `DecisionTree` representa el clasificador de árbol de decisiones y tiene los siguientes métodos y atributos:

- `maxDepth`: La profundidad máxima del árbol.
- `root`: El nodo raíz del árbol.

### Métodos

- `fit(X, y)`: Ajusta el árbol a los datos de entrenamiento.
- `buildTree(X, y, depth)`: Construye recursivamente el árbol de decisiones.
- `informationGain(parent, leftChild, rightChild)`: Calcula la ganancia de información de una división.
- `entropy(y)`: Calcula la entropía de un conjunto de etiquetas.
- `predict(X)`: Predice las etiquetas de clase para las muestras de entrada.

## Uso

Aquí hay un ejemplo básico de cómo usar el clasificador de árbol de decisiones:

```python
# Crear un clasificador de árbol de decisiones con una profundidad máxima de 5
tree = DecisionTree(maxDepth=5)

# Ajustar el árbol a los datos de entrenamiento
X_train = ...
y_train = ...
tree.fit(X_train, y_train)

# Realizar predicciones en nuevos datos
X_new = ...
predictions = tree.predict(X_new)
