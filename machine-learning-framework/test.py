
import numpy as np
from linear_regression import LinearRegressionModel


import pytest

class TestLinearRegressionModel:

    # Tests that the LinearRegressionModel object is initialized correctly
    def test_initialize_linear_regression_model(self):
        model = LinearRegressionModel()
        assert model.model_name == "Linear Regression"
        assert model.trained == False

    # Tests that the LinearRegression model is fitted to the training data successfully
    def test_fit_linear_regression_model(self):
        model = LinearRegressionModel()
        X_train = np.array([[1, 2, 3], [4, 5, 6]])
        y_train = np.array([1, 2])
        fitted_model = model.fit(X_train, y_train)
        assert fitted_model.trained == True

    # Tests that the LinearRegression model makes predictions on new data successfully
    def test_predict_linear_regression_model(self):
        model = LinearRegressionModel()
        X_train = np.array([[1, 2, 3], [4, 5, 6]])
        y_train = np.array([1, 2])
        model.fit(X_train, y_train)
        X_test = np.array([[7, 8, 9], [10, 11, 12]])
        predictions = model.predict(X_test)
        assert isinstance(predictions, np.ndarray)

    # Tests that the performance of the LinearRegression model is evaluated correctly
    def test_evaluate_linear_regression_model(self):
        model = LinearRegressionModel()
        X_train = np.array([[1, 2, 3], [4, 5, 6]])
        y_train = np.array([1, 2])
        model.fit(X_train, y_train)
        X_test = np.array([[7, 8, 9], [10, 11, 12]])
        y_test = np.array([3, 4])
        evaluation = model.evaluate(X_test, y_test)
        assert isinstance(evaluation, dict)
        assert 'Mean Squared Error' in evaluation
        assert 'R-squared (R^2)' in evaluation
        assert 'Mean Absolute Error (MAE)' in evaluation

    # Tests that an exception is raised when trying to predict without training the model
    def test_predict_without_training(self):
        model = LinearRegressionModel()
        X_test = np.array([[7, 8, 9], [10, 11, 12]])
        with pytest.raises(Exception):
            model.predict(X_test)

    # Tests that an exception is raised when trying to evaluate without training the model
    def test_evaluate_without_training(self):
        model = LinearRegressionModel()
        X_test = np.array([[7, 8, 9], [10, 11, 12]])
        y_test = np.array([3, 4])
        with pytest.raises(Exception):
            model.evaluate(X_test, y_test)