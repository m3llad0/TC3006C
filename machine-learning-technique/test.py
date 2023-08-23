from decision_tree import DecisionTree
import numpy as np
import pytest

class TestDecisionTree:

    # Tests that the fit method works correctly with valid input
    def test_fit_valid_input(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        maxDepth = 2
        dt = DecisionTree(maxDepth)
        dt.fit(X, y)
        assert dt.root is not None

    # Tests that the fit method raises a ValueError when given empty input
    def test_fit_empty_input(self):
        X = np.array([])
        y = np.array([])
        maxDepth = 2
        dt = DecisionTree(maxDepth)
        with pytest.raises(ValueError):
            dt.fit(X, y)

    # Tests that the fit method raises an IndexError when given invalid input
    def test_fit_invalid_input(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1, 0])
        maxDepth = 2
        dt = DecisionTree(maxDepth)
        with pytest.raises(IndexError):
            dt.fit(X, y)

    # Tests that the predict method works correctly with valid input
    def test_predict_valid_input(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        maxDepth = 2
        dt = DecisionTree(maxDepth)
        dt.fit(X, y)
        predictions = dt.predict(X)
        assert len(predictions) == len(y)

    # Tests that the predict method returns an empty list when given empty input
    def test_predict_empty_input(self):
        X = np.array([])
        maxDepth = 2
        dt = DecisionTree(maxDepth)
        predictions = dt.predict(X)
        assert predictions == []

    # Tests that the predict method does not raise a ValueError when given invalid input
    def test_predict_invalid_input(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        maxDepth = 2
        dt = DecisionTree(maxDepth)
        dt.fit(X, y)
        predictions = dt.predict(X)
        assert len(predictions) == len(y)