import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging


if __name__ == "__main__":
    logging.info("===== ML Pipeline Execution Started =====")
    print("\n===== ML Pipeline Execution Started =====")

    try:
        # -----------------------------------------------------
        # Data Ingestion
        # -----------------------------------------------------
        logging.info("Step 1: Initiating Data Ingestion...")
        print("Step 1: Initiating Data Ingestion...")

        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        logging.info("Data Ingestion completed successfully.")
        logging.debug(f"Train data path: {train_data_path}")
        logging.debug(f"Test data path: {test_data_path}")

        print(f"Train data path: {train_data_path}")
        print(f"Test data path: {test_data_path}\n")

        # -----------------------------------------------------
        # Data Transformation
        # -----------------------------------------------------
        logging.info("Step 2: Initiating Data Transformation...")
        print("Step 2: Initiating Data Transformation...")

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_file_path = data_transformation.initiate_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path
        )

        logging.info("Data Transformation completed successfully.")
        logging.debug(f"Train array shape: {train_arr.shape}")
        logging.debug(f"Test array shape: {test_arr.shape}")
        logging.debug(f"Preprocessor saved at: {preprocessor_file_path}")

        print(f"Train array shape: {train_arr.shape}")
        print(f"Test array shape: {test_arr.shape}")
        print(f"Preprocessor saved at: {preprocessor_file_path}\n")

        # -----------------------------------------------------
        # Model Training
        # -----------------------------------------------------
        logging.info("Step 3: Initiating Model Training...")
        print("Step 3: Initiating Model Training...")

        model_trainer = ModelTrainer()
        best_model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info("Model Training completed successfully.")
        logging.info(f"Best model R² score: {best_model_score:.4f}")

        print(f"Best model R² score: {best_model_score:.4f}\n")

    except Exception as e:
        logging.error("Pipeline execution failed due to an unexpected error.")
        print("Pipeline execution failed due to an unexpected error. Check logs for more details.")
        raise CustomException(e, sys)

    logging.info("===== ML Pipeline Execution Finished Successfully =====")
    print("===== ML Pipeline Execution Finished Successfully =====\n")
