"""
Data ingestion module for the ML project.

This component:
1. Reads the raw dataset.
2. Stores the raw data into the artifacts directory.
3. Splits the dataset into training and testing subsets.
4. Persists both splits in the artifacts folder.

It integrates with:
- CustomException for structured error reporting.
- Logging for operational transparency.
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

# -------------------------------------------------------------------------
# Configuration dataclass for ingestion paths
# -------------------------------------------------------------------------
@dataclass
class DataIngestionConfig:
    """
    Configuration paths for storing ingestion outputs.
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


# -------------------------------------------------------------------------
# Data Ingestion Component
# -------------------------------------------------------------------------
class DataIngestion:
    """
    Responsible for reading raw data, storing it, and performing
    train-test splitting.
    """

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Execute the data ingestion workflow:
        1. Read the dataset.
        2. Save raw data.
        3. Perform train-test split.
        4. Save split files.

        Returns
        -------
        tuple
            (train_data_path, test_data_path)
        """
        logging.info("Entered the DataIngestion component.")

        try:
            # STEP 1: Load raw dataset
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Dataset read into a DataFrame successfully.")

            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # STEP 2: Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw dataset saved at: {self.ingestion_config.raw_data_path}")

            # STEP 3: Split dataset
            logging.info("Initiating train-test split.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # STEP 4: Save split datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
