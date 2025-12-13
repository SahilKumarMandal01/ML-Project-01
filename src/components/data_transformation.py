"""
Data transformation module for the ML project.

This component:
1. Builds preprocessing pipelines for numerical and categorical features.
2. Applies transformations to training and test datasets.
3. Saves the fitted preprocessing object for later inference.

It integrates with:
- CustomException for structured error tracebacks.
- Logging for operational observability.
- Utility functions for serialization.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# -------------------------------------------------------------------------
# Configuration dataclass
# -------------------------------------------------------------------------
@dataclass
class DataTransformationConfig:
    """
    Stores the file path where the preprocessor object will be saved.
    """
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


# -------------------------------------------------------------------------
# Data Transformation Component
# -------------------------------------------------------------------------
class DataTransformation:
    """
    Responsible for building preprocessing pipelines and applying them
    to training and test datasets.
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # ---------------------------------------------------------------------
    def get_data_transformer_object(self):
        """
        Build and return a ColumnTransformer object that applies appropriate
        preprocessing to numerical and categorical features.

        Returns
        -------
        ColumnTransformer
            Preprocessing pipeline for model-ready data.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical feature pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical feature pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Combine feature pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------------------------------------------------
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Apply preprocessing pipelines to training and test datasets.

        Parameters
        ----------
        train_path : str
            Path to training CSV file.
        test_path : str
            Path to test CSV file.

        Returns
        -------
        tuple
            (train_array, test_array, preprocessor_file_path)
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded training and testing datasets successfully.")
            logging.info("Obtaining preprocessing object...")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing pipelines to training and test datasets."
            )

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Combine transformed features and target
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Preprocessing completed. Saving preprocessor object.")

            # Save preprocessor object for inference pipeline
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
