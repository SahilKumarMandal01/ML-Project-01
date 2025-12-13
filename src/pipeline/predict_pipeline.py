"""
Prediction pipeline module.

This module provides:
1. PredictPipeline — loads artifacts and generates predictions.
2. CustomData — converts user input into a DataFrame compatible with the trained model.

Works with:
- Serialized model and preprocessor stored in artifacts/
- CustomException for clean traceback handling
"""

import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


# -------------------------------------------------------------------------
# Prediction Pipeline
# -------------------------------------------------------------------------
class PredictPipeline:
    """
    Load preprocessing artifacts and models, transform incoming data,
    and return predictions.
    """

    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        Generate predictions using stored preprocessor and model.

        Parameters
        ----------
        features : pd.DataFrame
            Input features in a DataFrame format.

        Returns
        -------
        np.ndarray
            Array of predictions.
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Loading model and preprocessor...")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("Artifacts loaded successfully.")

            transformed_data = preprocessor.transform(features)
            predictions = model.predict(transformed_data)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


# -------------------------------------------------------------------------
# Custom Data Wrapper
# -------------------------------------------------------------------------
class CustomData:
    """
    Convert raw user input into a DataFrame suitable for prediction.
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Convert stored attributes into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame containing all required model inputs.
        """
        try:
            input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(input_dict)

        except Exception as e:
            raise CustomException(e, sys)
