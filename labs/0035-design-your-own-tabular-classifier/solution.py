# --- cell 1 ---
import numpy as np
import pandas as pd
# You may import numpy, pandas, scipy, and sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

class AdultCleaner(BaseEstimator, TransformerMixin):
    """Normalizes raw census strings, missing tokens, and skewed numeric columns."""

    def __init__(self):
        self.num_cols_ = []
        self.cat_cols_ = []

    def fit(self, X, y=None):
        df = self._clean_raw(X)
        self.num_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = [c for c in df.columns if c not in self.num_cols_]
        return self
    def transform(self, X):
        df = self._clean_raw(X)

        # Ensure expected schema and dtypes at inference time
        for col in self.num_cols_:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Heavy-tailed capital gain/loss columns benefit from log1p + non-zero flags
            lower_name = col.lower().replace("_", "-")
            if "capital" in lower_name:
                df[f"{col}_has_val"] = (df[col].fillna(0) > 0).astype(float)
                df[col] = np.log1p(np.maximum(df[col], 0))

        for col in self.cat_cols_:
            df[col] = df[col].astype("object")
            df[col] = df[col].where(df[col].notna(), np.nan)

        return df
    @staticmethod
    def _clean_raw(X):
        df = X.copy()
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or isinstance(
                df[col].dtype, pd.CategoricalDtype
            ):
                s = df[col].astype("string").str.strip()
                # Adult dataset often encodes missing values as '?'
                s = s.replace({"?": pd.NA, "": pd.NA, "nan": pd.NA, "None": pd.NA})
                df[col] = s.astype("object")
        return df
    

def _build_tree_preprocessor(num_cols, cat_cols):
    """Ordinal encoding with native NaN support for HistGradientBoosting."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    
def _build_linear_preprocessor(num_cols, cat_cols):
    """One-hot encoding + robust scaling for LogisticRegression / bagging trees."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    
    

                        

def train(X_train, y_train, X_val, y_val):
    """
    Design your own tabular binary classifier on real census data.

    X_* are pandas DataFrames with MIXED dtypes (numeric + categorical).
    Features are RAW — you own imputation, encoding, scaling, imbalance
    handling, model choice, and calibration.

    Target: predict whether income is >50K (class 1, minority).

    Args:
        X_train: pd.DataFrame (n_train, n_features)
        y_train: np.ndarray (n_train,) binary labels {0, 1}
        X_val:   pd.DataFrame (n_val, n_features)
        y_val:   np.ndarray (n_val,) binary labels {0, 1}

    Returns:
        predict_proba: callable predict_proba(X) -> np.ndarray (n,)
            Positive-class scores for a DataFrame X with the same
            columns as X_train. Finite scores; ranking is what matters
            for PR-AUC.
    """
    # TODO: build any classical pipeline you like
    # Ideas:
    #   - ColumnTransformer: impute + scale numerics, encode categoricals
    #   - class_weight, resampling / SMOTE on the train fold only
    #   - logistic / SVM / RF / HistGradientBoosting / stacking
    #   - tune on val, refit on train+val
    #   - calibrate on val
    y_train = np.asarray(y_train, dtype=int).ravel()
    y_val = np.asarray(y_val, dtype=int).ravel()
    cleaner = AdultCleaner()
    X_tr_clean = cleaner.fit_transform(X_train)
    X_va_clean = cleaner.transform(X_val)
    num_cols = cleaner.num_cols_
    cat_cols = cleaner.cat_cols_
    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    
    hgb_configs = [
        {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 30, "l2_regularization": 1.0},
        {"learning_rate": 0.03, "max_leaf_nodes": 63, "min_samples_leaf": 40, "l2_regularization": 5.0},
        {"learning_rate": 0.07, "max_leaf_nodes": 23, "min_samples_leaf": 20, "l2_regularization": 0.5},
    ]

    best_hgb_params = hgb_configs[0]
    best_hgb_score = -1.0
    for params in hgb_configs:
        pipe = Pipeline(
            steps=[
                ("prep", _build_tree_preprocessor(num_cols, cat_cols)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=400,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=25,
                        scoring="average_precision",
                        categorical_features=cat_indices if cat_indices else None,
                        random_state=42,
                        **params,
                    ),
                ),
            ]
        )
        pipe.fit(X_tr_clean, y_train)
        val_preds = pipe.predict_proba(X_va_clean)[:, 1]
        score = average_precision_score(y_val, val_preds)
        if score > best_hgb_score:
            best_hgb_score = score
            best_hgb_params = params
    
    best_lr_c = 1.0
    best_lr_score = -1.0

    for c_val in [0.2, 1.0, 3.0]:
        lr_pipe = Pipeline(
            steps=[
                ("prep", _build_linear_preprocessor(num_cols, cat_cols)),
                (
                    "clf",
                    LogisticRegression(
                        C=c_val,
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
        lr_pipe.fit(X_tr_clean, y_train)
        val_preds = lr_pipe.predict_proba(X_va_clean)[:, 1]
        score = average_precision_score(y_val, val_preds)
        if score > best_lr_score:
            best_lr_score = score
            best_lr_c = c_val
    X_full = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_full = np.concatenate([y_train, y_val], axis=0)

    final_cleaner = AdultCleaner()
    X_full_clean = final_cleaner.fit_transform(X_full)
    full_num = final_cleaner.num_cols_
    full_cat = final_cleaner.cat_cols_
    full_cat_idx = list(range(len(full_num), len(full_num) + len(full_cat)))

    hgb_final = Pipeline(
        steps=[
            ("prep", _build_tree_preprocessor(full_num, full_cat)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=25,
                    scoring="average_precision",
                    categorical_features=full_cat_idx if full_cat_idx else None,
                    random_state=42,
                    **best_hgb_params,
                ),
            ),
        ]
    )        
    
    hgb_alt_final = Pipeline(
        steps=[
            ("prep", _build_tree_preprocessor(full_num, full_cat)),
            (
                 "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_leaf_nodes=45,
                    min_samples_leaf=35,
                    l2_regularization=2.0,
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=25,
                    scoring="average_precision",
                    categorical_features=full_cat_idx if full_cat_idx else None,
                    random_state=1337,
                ),
            ),
        ]
    )

    rf_final = Pipeline(
        steps=[
            ("prep", _build_linear_preprocessor(full_num, full_cat)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=18,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    
    
    lr_final = Pipeline(
        steps=[
            ("prep", _build_linear_preprocessor(full_num, full_cat)),
            (
                "clf",
                LogisticRegression(
                    C=best_lr_c,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    
    ensemble = VotingClassifier(
        estimators=[
            ("hgb_tuned", hgb_final),
            ("hgb_alt", hgb_alt_final),
            ("rf", rf_final),
            ("lr", lr_final),
        ],
        voting="soft",
        weights=[3.0, 2.0, 1.0, 1.0],
    )
    ensemble.fit(X_full_clean, y_full)
    
    def predict_proba(X):
        X_clean = final_cleaner.transform(X)
        probs = ensemble.predict_proba(X_clean)[:, 1]
        # Guarantee finite 1D numpy float array
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        return np.asarray(probs, dtype=np.float64).ravel()


    return predict_proba