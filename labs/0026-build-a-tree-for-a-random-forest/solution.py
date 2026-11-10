# --- cell 1 ---
import numpy as np

class Node:
    """Internal container representing a single structural node or leaf within the tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Index of the feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Predicted class (only if it is a leaf node)

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
    A decision tree classifier that the harness will use as the base learner
    in a Random Forest. The harness will create many DecisionTree instances,
    train each on a different bootstrap sample of the data, and aggregate
    their predictions by majority vote.

    For the ensemble to beat a single tree by a meaningful margin, your trees
    must be DIVERSE. Bootstrap sampling (handled by the harness) gives some
    diversity. The most effective additional source of diversity is to
    randomize WHICH features each split considers -- this is what makes a
    Random Forest different from plain Bagging.

    Use `self.random_state` for any randomness inside your tree so that
    different harness seeds produce different trees.
    """

    def __init__(self, max_depth=10, min_samples_split=2,
                 max_features='sqrt', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features      # 'sqrt', None, or int
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        """
        Fit the tree on (X, y).

        Args:
            X: numpy array of shape (n_samples, n_features), dtype float
            y: numpy array of shape (n_samples,), integer class labels in [0, n_classes)

        Returns:
            self
        """
        # TODO: build the tree, storing it on self
        self._rng = np.random.default_rng(self.random_state)
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        unique_classes, counts = np.unique(y, return_counts=True)
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(unique_classes) == 1):
            leaf_value = unique_classes[np.argmax(counts)]
            return Node(value=leaf_value)
        if self.max_features == 'sqrt':
            n_sub = int(np.sqrt(n_features))
        elif self.max_features is None:
            n_sub = n_features
        else:
            n_sub = int(self.max_features)
        n_sub = max(1, min(n_sub, n_features))
        feature_indices = self._rng.choice(n_features, size=n_sub, replace=False)
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        current_gini = self._gini_impurity(y)
        for feat_idx in feature_indices:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                left_mask = X_column <= threshold
                y_left = y[left_mask]
                y_right = y[~left_mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                w_left = len(y_left) / n_samples
                w_right = len(y_right) / n_samples
                child_gini = w_left * self._gini_impurity(y_left) + w_right * self._gini_impurity(y_right)
                gain = current_gini - child_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold
                    
        if best_gain <= 0.0 or best_feature is None:
            leaf_value = unique_classes[np.argmax(counts)]
            return Node(value=leaf_value)
            
        left_mask = X[:, best_feature] <= best_threshold
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
            
    def _gini_impurity(self, y):
        """Calculates standard Gini Impurity: 1 - sum(p_i^2)."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def predict(self, X):
        """
        Predict integer class labels for X.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with integer class labels
        """
        # TODO: traverse the tree for each row in X
        return np.array([self._traverse_tree(row, self.root) for row in X])
    
    def _traverse_tree(self, row, node):
        """Navigates down the structural graph configuration for a single input row."""
        if node.is_leaf:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._traverse_tree(row, node.left)
        return self._traverse_tree(row, node.right)