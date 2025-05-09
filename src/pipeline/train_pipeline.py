import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        logging.info("Training pipeline initialized")
    
    def run_pipeline(self):
        """
        Run the complete training pipeline:
        1. Data Ingestion
        2. Data Transformation
        3. Model Training
        """
        try:
            # Data Ingestion
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            # Data Transformation
            logging.info("Starting data transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            # Model Training
            logging.info("Starting model training")
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.info(f"Training pipeline completed with R2 score: {r2_score}")
            return r2_score
            
        except Exception as e:
            logging.error("Error in training pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Execute the training pipeline
    pipeline = TrainPipeline()
    r2_score = pipeline.run_pipeline()
    print(f"Final model R2 score: {r2_score}")