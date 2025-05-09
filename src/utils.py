import os
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load an object from a file using dill
    
    Args:
        file_path (str): Path to the saved object
        
    Returns:
        object: The loaded object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
        
        

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            # Check if the model has hyperparameters defined
            param_grid = params.get(model_name, {})

            try:
                if param_grid:
                    logging.info(f"Applying GridSearchCV for {model_name}")
                    gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1, error_score='raise')
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                else:
                    logging.info(f"No hyperparameter tuning for {model_name}")
                    model.fit(X_train, y_train)
                    best_model = model

                # Predictions
                y_test_pred = best_model.predict(X_test)

                # R^2 score
                test_score = r2_score(y_test, y_test_pred)
                report[model_name] = test_score
                
            except Exception as e:
                logging.warning(f"Error training {model_name}: {e}")
                # Skip this model but continue with others
                report[model_name] = float('-inf')  # Use negative infinity so it won't be selected as best

        return report

    except Exception as e:
        raise CustomException(e, sys)
