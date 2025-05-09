import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# ModelTrainerConfig holds the path where the trained model will be saved (artifacts/model.pkl).
# @dataclass simplifies class creation by auto-generating init methods.

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, allow_writing_files=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Random Forest Regressor": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10, 20]
                },
                "XGBRegressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [100, 200]
                },
                "GradientBoostingRegressor": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50, 100]
                }
            }

            logging.info("Starting model evaluation with hyperparameter tuning")
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Print all model scores for comparison
            print("\nModel Performance Summary:")
            for model_name, score in model_report.items():
                print(f"{model_name}: {score:.4f}")
            print()

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f"Best model: {best_model_name} with score: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No suitable model found (R2 < 0.6)", sys)

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
