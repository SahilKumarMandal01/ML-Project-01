"""
Optimized model training module for faster pipeline testing.

Key Features
------------
1. `fast_mode` dramatically reduces training time via:
    - Smaller hyperparameter search spaces.
    - Use of RandomizedSearchCV instead of GridSearchCV.
    - Optional exclusion of heavy models (XGBoost, CatBoost).

2. Preserves:
    - Model selection logic
    - R² scoring on test data
    - Artifact saving mechanism

Intended Use
------------
Use this module during experimentation, debugging, and CI/CD pipeline tests
to keep execution time minimal while preserving workflow correctness.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# -------------------------------------------------------------------------
# Configuration Dataclass
# -------------------------------------------------------------------------
@dataclass
class ModelTrainerConfig:
    """
    Stores the file path where the trained model will be saved.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


# -------------------------------------------------------------------------
# Optimized Model Trainer Component
# -------------------------------------------------------------------------
class ModelTrainer:
    """
    Train, evaluate, and select the best regression model using either fast mode
    (reduced training time) or standard mode (more exhaustive search).

    Parameters
    ----------
    fast_mode : bool
        When True:
            - Uses reduced hyperparameter grids.
            - Trains lightweight models.
            - Performs limited RandomizedSearch iterations.
        When False:
            - Includes heavy models (XGBoost, CatBoost).
            - Uses larger search spaces.
            - Trains slower but potentially better models.
    """

    def __init__(self, fast_mode: bool = True):
        self.config = ModelTrainerConfig()
        self.fast_mode = fast_mode

    # ---------------------------------------------------------------------
    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        """
        Execute the complete training, evaluation, and model selection workflow.

        Parameters
        ----------
        train_array : np.ndarray
            Training data where the last column represents the target.
        test_array : np.ndarray
            Test data where the last column represents the target.

        Returns
        -------
        float
            Best model R² score on the test dataset.
        """
        try:
            logging.info("Splitting arrays into X/y sets...")

            # Split input and target arrays
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # ===============================================================
            # Define Models
            # ===============================================================
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
            }

            # Include heavy models only when not in fast mode
            if not self.fast_mode:
                models.update({
                    "XGBRegressor": XGBRegressor(tree_method="hist", eval_metric="rmse", verbosity=0),
                    "CatBoost Regressor": CatBoostRegressor(verbose=False),
                })

            # ===============================================================
            # Hyperparameter Search Grids
            # ===============================================================
            if self.fast_mode:
                params = {
                    "Decision Tree": {
                        "max_depth": [3, 5, 7],
                        "criterion": ["squared_error", "friedman_mse"],
                    },
                    "Random Forest": {
                        "n_estimators": [10, 30, 50],
                        "max_depth": [5, 7],
                    },
                    "Gradient Boosting": {
                        "n_estimators": [20, 40],
                        "learning_rate": [0.1, 0.05],
                    },
                    "AdaBoost": {
                        "n_estimators": [20, 50],
                        "learning_rate": [0.1, 0.05],
                    },
                    "Linear Regression": {}
                }
            else:
                params = {
                    "Decision Tree": {
                        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                        "max_depth": [3, 5, 7, 10],
                    },
                    "Random Forest": {
                        "n_estimators": [50, 100, 150, 200],
                        "max_depth": [5, 7, 10, 12],
                    },
                    "Gradient Boosting": {
                        "n_estimators": [50, 100, 150],
                        "learning_rate": [0.1, 0.05, 0.01],
                    },
                    "AdaBoost": {
                        "n_estimators": [50, 100, 150],
                        "learning_rate": [0.1, 0.05],
                    },
                    "XGBRegressor": {
                        "n_estimators": [50, 100, 150],
                        "learning_rate": [0.1, 0.05],
                    },
                    "CatBoost Regressor": {
                        "learning_rate": [0.1, 0.05],
                        "depth": [6, 8, 10],
                        "iterations": [100, 200],
                    },
                }

            # ===============================================================
            # Hyperparameter Optimization (RandomizedSearchCV)
            # ===============================================================
            logging.info("Starting model evaluation using RandomizedSearchCV...")

            model_report = {}
            best_models = {}

            for model_name, model in models.items():
                param_grid = params.get(model_name, {})

                # Randomized Search if params exist, else direct training
                if param_grid:
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grid,
                        n_iter=5 if self.fast_mode else 15,
                        cv=3,
                        n_jobs=-1,
                        random_state=42,
                        verbose=0,
                    )
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    best_model = model

                preds = best_model.predict(X_test)
                score = r2_score(y_test, preds)

                model_report[model_name] = score
                best_models[model_name] = best_model

                logging.info(f"{model_name} | R² Score: {score:.4f}")

            # ===============================================================
            # Select Best Model
            # ===============================================================
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = best_models[best_model_name]

            logging.info(f"Best Model Selected: {best_model_name} | Score: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No model achieved acceptable performance.")

            # ===============================================================
            # Save Best Model
            # ===============================================================
            save_object(self.config.trained_model_file_path, best_model)
            logging.info(f"Model saved at: {self.config.trained_model_file_path}")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
