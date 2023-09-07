from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from typing import Dict

class LinearRegressionModel:
    """
    A class that implements linear regression using the LinearRegression model from the sklearn.linear_model module.
    
    Methods:
    - __init__(): Initializes the LinearRegressionModel object.
    - fit(X_train, y_train): Fits the LinearRegression model to the training data.
    - predict(X_test): Makes predictions using the trained LinearRegression model on new data.
    - evaluate(X_test, y_test): Evaluates the performance of the trained LinearRegression model.
    
    Fields:
    - model: An instance of the LinearRegression model from the sklearn.linear_model module.
    - model_name: A string representing the name of the model.
    - trained: A boolean indicating whether the model has been trained or not.
    """
    
    def __init__(self) -> None:
        """
        Initializes the LinearRegressionModel object.
        """
        self.model = LinearRegression()
        self.model_name = "Linear Regression"
        self.trained = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'LinearRegressionModel':
        """
        Fits the LinearRegression model to the training data.
        
        Parameters:
        - X_train: The input features of the training data.
        - y_train: The target values of the training data.
        
        Returns:
        - The LinearRegressionModel object.
        """
        self.model.fit(X_train, y_train)
        self.trained = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained LinearRegression model on new data.
        
        Parameters:
        - X_test: The input features of the test data.
        
        Returns:
        - The predicted target values.
        
        Raises:
        - Exception: If the model has not been trained yet.
        """
        if not self.trained:
            raise Exception("Model not trained yet")
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluates the performance of the trained LinearRegression model.
        
        Parameters:
        - X_test: The input features of the test data.
        - y_test: The target values of the test data.
        
        Returns:
        - A dictionary containing the evaluation metrics: Model, MSE, RMSE, MAE, R2.
        
        Raises:
        - Exception: If the model has not been trained yet.
        """
        if not self.trained:
            raise ValueError("El modelo no ha sido entrenado. Debes llamar al método 'fit' primero.")
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        return {
            'Mean Squared Error': mse,
            'R-squared (R^2)': r2,
            'Mean Absolute Error (MAE)': mae
        }