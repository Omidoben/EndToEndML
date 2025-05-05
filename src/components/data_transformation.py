import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object           # utility function to save the preprocessing object as a .pkl file

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts", "preprocessor.pkl")

    # This is a configuration class that stores the path where the preprocessor object will be saved (as preprocessor.pkl inside an artifacts/ folder)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()          # The constructor initializes an instance of DataTransformationConfig
    
    def get_data_transformer_object(self):

        """This method handles data transformation. It builds the transformation logic"""

        try:
            numerical_columns= ["writing_score", "reading_score"]
            categorical_columns= [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
                ]

            num_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

        
    def initiate_data_transformation(self, train_path, test_path):

        """This method applies the transformation logic to the datasets."""

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocesing object")

            preprocessing_obj=self.get_data_transformer_object()      # Gets the preprocessing object we  built above

            target_column_name="math_score"

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data frames")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)       # learns and applies transformation
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)             # applies the same transformation without re-learning

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]           # Concatenates the transformed features and the target into one array for each set
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            # Saves the preprocessing_obj as a .pkl file so it can be reused during prediction/inference

            return(
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            # Returns: Transformed training array, testing array, and Path to the saved preprocessor object

        except Exception as e:
            raise CustomException(e, sys)


