# --- cell 1 ---
import math
import random

def train(X_train, y_train, X_val, y_val, n_classes):
    """
    Train a digit classifier using only Python and the math module.

    The dataset contains 8x8 pixel images of digits 0-9, flattened to 64 features.
    Pixel values are normalized to [0, 1].

    Args:
        X_train: list[list[float]] -- training images (each 64 floats)
        y_train: list[int]         -- training labels (each in [0, 10))
        X_val:   list[list[float]] -- validation images
        y_val:   list[int]         -- validation labels
        n_classes: int             -- number of classes (10)

    Returns:
        predict: callable, list[list[float]] -> list[int]
            Takes a list of images, returns a list of predicted class labels.
    """
    # TODO: implement your classifier
    # You may use ANY approach: centroid, KNN, logistic regression,
    # neural net with backprop, decision tree, or anything creative.
    def predict_with_ref(X_queries, ref_X, ref_y, k):
        predictions = []
        for x in X_queries:
            dists = []
            for idx, rx in enumerate(ref_X):
                dist_sq = sum((a - b) * (a - b) for a, b in zip(x, rx))
                dists.append((dist_sq, ref_y[idx]))
            dists.sort(key=lambda pair: pair)
            neighbors = dists[:k]
            class_weights = [0.0] * n_classes
            for dist_sq, label in neighbors:
                weight = 1.0 / (math.sqrt(dist_sq) + 1e-8)
                class_weights[label] += weight
            best_class = max(range(n_classes), key=lambda c: class_weights[c])
            predictions.append(best_class)
        return predictions
    best_k = 3
    best_acc = -1.0
    if X_val and y_val:
        for k in (1, 3, 5, 7):
            val_preds = predict_with_ref(X_val, X_train, y_train, k)
            correct = sum(1 for p, y in zip(val_preds, y_val) if p == y)
            acc = correct / len(y_val)
            if acc > best_acc:
                best_acc = acc
                best_k = k
    full_X = list(X_train) + (list(X_val) if X_val else [])
    full_y = list(y_train) + (list(y_val) if y_val else [])

    def predict(X_test):
        return predict_with_ref(X_test, full_X, full_y, best_k)

    return predict