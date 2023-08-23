import numpy as np

class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        featureIndex (int): The index of the feature used for splitting at this node.
        threshold (float): The threshold value used for splitting at this node.
        left (Node): The left child node.
        right (Node): The right child node.
        value (str): The value of the label if this node is a leaf node.
    """
    def __init__(self, featureIndex=None, threshold=None, left=None, right=None, value=None):
        """
        Initializes a new instance of the Node class.

        Args:
            featureIndex (int, optional): The index of the feature used for splitting at this node. Defaults to None.
            threshold (float, optional): The threshold value used for splitting at this node. Defaults to None.
            left (Node, optional): The left child node. Defaults to None.
            right (Node, optional): The right child node. Defaults to None.
            value (str, optional): The value of the label if this node is a leaf node. Defaults to None.
        """
        self.featureIndex = featureIndex
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    """
    A class representing a decision tree classifier.

    Parameters:
    -----------
    maxDepth : int
        The maximum depth of the decision tree.

    Attributes:
    -----------
    maxDepth : int
        The maximum depth of the decision tree.
    root : Node
        The root node of the decision tree.

    Methods:
    -----------
    fit(X, y)
        Fit the decision tree to the training data.
    buildTree(X, y, depth)
        Recursively build the decision tree.
    informationGain(parent, leftChild, rightChild)
        Calculate the information gain of a split.
    entropy(y)
        Calculate the entropy of a set of labels.
    predict(X)
        Predict the class labels for the input samples.
    predictionTree(node, sample)
        Recursively traverse the decision tree to make predictions.
    """

    def __init__(self, maxDepth):
        self.maxDepth = maxDepth
        self.root = None
    
    def fit(self, X, y):
        """
        Fit the decision tree to the training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns:
        -----------
        None
        """
        self.root = self.buildTree(X, y, depth=0)
    
    def buildTree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values.
        depth : int
            The current depth of the tree.

        Returns:
        -----------
        Node
            The root node of the decision tree.
        """
        numSamples, numFeatures = X.shape
        numClasses = len(np.unique(y))

        # Stop building the tree if the maximum depth is reached or if there is only one class
        if depth == self.maxDepth or numClasses == 1:
            return Node(value=np.bincount(y).argmax())

        # Find the best threshold to split the data

        bestGain = 0.0
        bestFeatureIndex = None
        bestThreshold = None

        for featureIndex in range(numFeatures):
            thresholds = np.unique(X[:, featureIndex])
            for threshold in thresholds:
                gain = self.informationGain(y, y[X[:, featureIndex] <= threshold], y[X[:, featureIndex] > threshold])

                if gain > bestGain:
                    bestGain = gain
                    bestFeatureIndex = featureIndex
                    bestThreshold = threshold

        if bestGain > 0:
            leftIndex = X[:, bestFeatureIndex] <= bestThreshold
            rightIndex = ~leftIndex
            leftSubtree = self.buildTree(X[leftIndex], y[leftIndex], depth=depth + 1)
            rightSubtree = self.buildTree(X[rightIndex], y[rightIndex], depth=depth + 1)
            return Node(bestFeatureIndex, bestThreshold, leftSubtree, rightSubtree)
        else:
            return Node(value=np.bincount(y).argmax())
        
    def informationGain(self, parent, leftChild, rightChild):
        """
        Calculate the information gain of a split.

        Parameters:
        -----------
        parent : array-like, shape (n_samples,)
            The labels of the parent node.
        leftChild : array-like, shape (n_samples,)
            The labels of the left child node.
        rightChild : array-like, shape (n_samples,)
            The labels of the right child node.

        Returns:
        -----------
        float
            The information gain of the split.
        """
        numParent = len(parent)
        numLeft = len(leftChild)
        numRight = len(rightChild)

        entropyParent = self.entropy(parent)
        entropyChildren = (numLeft / numParent) * self.entropy(leftChild) + (numRight / numParent) * self.entropy(rightChild)
        informationGain = entropyParent - entropyChildren
        return informationGain
    
    def entropy(self, y):
        """
        Calculate the entropy of a set of labels.

        Parameters:
        -----------
        y : array-like, shape (n_samples,)
            The labels.

        Returns:
        -----------
        float
            The entropy of the labels.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns:
        -----------
        list
            The predicted class labels.
        """
        if X.size == 0:
            return []
        
        predictions = [self.predictionTree(self.root, sample) for sample in X]
        return predictions

    def predictionTree(self, node, sample):
        """
        Recursively traverse the decision tree to make predictions.

        Parameters:
        -----------
        node : Node
            The current node in the decision tree.
        sample : array-like, shape (n_features,)
            The input sample.

        Returns:
        -----------
        int
            The predicted class label.
        """
        if node.value is not None:
            return node.value
        if sample[node.featureIndex] <= node.threshold:
            return self.predictionTree(node.left, sample)
        else:
            return self.predictionTree(node.right, sample)
        