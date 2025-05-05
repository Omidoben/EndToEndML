import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts", "train.csv")
    test_data_path: str=os.path.join("artifacts", "test.csv")
    raw_data_path: str=os.path.join("artifacts", "data.csv")

# Explanation of above code:
#   - This creates a configuration class that holds default file paths for storing: Raw data (data.csv), 
#     Training data (train.csv), Testing data (test.csv)

# - These will be stored in a folder called artifacts.
# - The @dataclass automatically creates the __init__ method, we use it because we're only defining variables



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()  
# Initializes a new object with self.ingestion_config, giving access to all the file paths defined earlier.

    def initiate_data_ingestion(self):                                      # main method to ingest data
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the dataset as a data frame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)      # Ensures that the artifacts/ directory exists; creates it if not

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)                # Saves the raw DataFrame to data.csv.

            logging.info("Train test split initiated")
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)        # Saves the training and testing sets to their respective paths.

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,                  # Returns the paths so they can be used by the next pipeline stage (like data transformation or model training).


                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _=data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))




# Explanation of above code

# - This is a Python standard construct that ensures the code inside this block only runs when the script is executed directly, 
#   not when it's imported as a module in another file

# obj - Creates an instance of the DataIngestion class, which is responsible for reading the raw data, splitting it, 
#       saving them to disk, and returning the file paths to the csv files

# train_data, test_data - Calls the initiate_data_ingestion() method, which does the ingestion work, and stores the returned file paths to train_data and test_data.
#                         train_data - is a file path to the training data CSV (e.g., "artifacts/train.csv")
#                         test_data - is a file path to the test data CSV (e.g., "artifacts/test.csv")

# data_transformation - creates an instance of the DataTransformation class
# Last line applies our previously defined data preprocessing pipeline to the train and test data