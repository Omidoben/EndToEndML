import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class PredictionPipelineConfig:
    model_path = os.path.join("artifacts", "model.pkl")
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

class PredictPipeline:
    def __init__(self):
        self.prediction_config = PredictionPipelineConfig()

    def predict(self, features):
        """
        Make predictions using the trained model and preprocessing pipeline

        Args:
            features (pd.DatFrame): Input features for prediction

        Returns:
            np.ndarray: Predicted values
        """
        try:
            # Load the preprocessor and model
            model_path = self.prediction_config.model_path
            preprocessor_path = self.prediction_config.preprocessor_path

            logging.info("Loading model and preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform the input features
            data_scaled = preprocessor.transform(features)

            # Make predictions
            predictions = model.predict(data_scaled)
            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Class to convert user input to a dataframe for prediction
    """
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
        ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        """
        Convert the user input data to a data frame

        Returns:
            pd.DataFrame: Input data as a data frame
        """
        try:
            input_data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(input_data)

        except Exception as e:
            raise CustomException(e, sys)


    
# Example of how it can be used

if __name__=="__main__":
    # sample input data
    sample_data=CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="completed",
        reading_score=72,
        writing_score=74
    )

    # Convert to DataFrame
    input_df = sample_data.get_data_as_dataframe()
    print("Input DataFrame:\n", input_df)

    # Initialize the prediction pipeline
    predict_pipeline = PredictPipeline()
    
    # Make prediction
    try:
        prediction = predict_pipeline.predict(input_df)
        print(f"Predicted math score: {prediction[0]:.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

