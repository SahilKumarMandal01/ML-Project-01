"""
Utility functions for model persistence, loading, and evaluation.

This module provides:
1. save_object   — serialize a Python object (e.g., model, preprocessor).
2. load_object   — load serialized objects.
3. evaluate_models — perform GridSearchCV hyperparameter tuning and compute model scores.

Integrated with:
- CustomException for standardized error handling.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


# -------------------------------------------------------------------------
def save_object(file_path: str, obj) -> None:
    """
    Serialize and save a Python object to the specified file path.

    Parameters
    ----------
    file_path : str
        Destination path where the object will be stored.
    obj : Any
        Python object to persist (model, preprocessor, transformer, etc.)
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# -------------------------------------------------------------------------
def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models: dict,
    param: dict,
) -> dict:
    """
    Train and evaluate multiple models using hyperparameter tuning (GridSearchCV).

    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training input features.
    y_train : np.ndarray or pd.Series
        Training target values.
    X_test : np.ndarray or pd.DataFrame
        Test input features.
    y_test : np.ndarray or pd.Series
        Test target values.
    models : dict
        Dictionary of model_name → model_instance.
    param : dict
        Dictionary of model_name → parameter_grid for GridSearchCV.

    Returns
    -------
    dict
        Dictionary mapping model_name → test_r2_score.
    """
    try:
        report = {}

        model_names = list(models.keys())
        model_objects = list(models.values())

        for idx, model_name in enumerate(model_names):
            model = model_objects[idx]
            param_grid = param[model_name]

            # Hyperparameter tuning
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            # Set the best parameters and retrain
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Performance metrics
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


# -------------------------------------------------------------------------
def load_object(file_path: str):
    """
    Load and deserialize a Python object from disk.

    Parameters
    ----------
    file_path : str
        Path to the serialized object.

    Returns
    -------
    Any
        Deserialized Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
