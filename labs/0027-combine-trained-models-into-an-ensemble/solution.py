# --- cell 1 ---
import numpy as np
# You can import any sklearn module you need
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_ensemble(base_models, X_val, y_val):
    """
    Combine K pre-trained base classifiers into an ensemble that predicts class
    labels on new data.

    The harness has already trained the base models on a separate training set.
    You DO NOT retrain them. Instead, use their predictions on (X_val, y_val) to
    figure out how to combine them, then return a `predict` callable that the
    harness will apply to held-out test data.

    Args:
        base_models: list of K trained sklearn-style classifiers. Each has:
            - .predict(X) -> np.ndarray of integer class labels, shape (n,)
            - .predict_proba(X) -> np.ndarray of shape (n, n_classes)
        X_val: numpy array of shape (n_val, n_features) -- standardized
        y_val: numpy array of shape (n_val,) -- integer class labels

    Returns:
        predict: callable taking X (n, n_features) and returning a numpy array
                 of integer class labels, shape (n,)
    """
    # TODO: design a combiner. Some options to consider:
    #   * Hard voting:     majority of base_models[k].predict(X) across k
    #   * Soft voting:     argmax of mean predict_proba across base models
    #   * Weighted voting: weight each model by its validation accuracy
    #   * Stacking:        train a meta-model on base predict_proba outputs
    # The win threshold is set so that naive voting strategies will not pass.
    eps = 1e-12

    def build_meta_features(X):
        probas = [m.predict_proba(X) for m in base_models]
        concat_probas = np.hstack(probas)
        log_probas = np.log(np.clip(concat_probas, eps, 1.0))
        entropies = []
        margins = []
        for p in probas:
            p_clipped = np.clip(p, eps, 1.0)
            ent = -np.sum(p_clipped * np.log(p_clipped), axis=1, keepdims=True)
            sorted_p = np.sort(p, axis=1)
            margin = (sorted_p[:, -1] - sorted_p[:, -2])[:, None]
            entropies.append(ent)
            margins.append(margin)
        preds = [m.predict(X) for m in base_models]
        disagreements = []
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                disagreements.append((preds[i] != preds[j]).astype(float)[:, None])
        return np.hstack([concat_probas, log_probas] + entropies + margins + disagreements)
    Z_val = build_meta_features(X_val)
    scaler = StandardScaler()
    Z_val_scaled = scaler.fit_transform(Z_val)
    meta_clf = LogisticRegression(
        C=0.5,
        max_iter=2000,
        random_state=42,
    )
    meta_clf.fit(Z_val_scaled, y_val)

    def predict(X):
        # TODO: compute and return integer class predictions for X
        Z = build_meta_features(X)
        Z_scaled = scaler.transform(Z)
        return meta_clf.predict(Z_scaled).astype(int)
    
    return predict