from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging
import os
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            return render_template('form.html')
        else:
            # Get data from form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            
            # Convert to DataFrame
            input_df = data.get_data_as_dataframe()
            
            # Initialize prediction pipeline
            predict_pipeline = PredictPipeline()
            
            # Make prediction
            results = predict_pipeline.predict(input_df)
            
            return render_template('results.html', prediction=results[0])
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        # Get data from JSON request
        json_data = request.json
        
        data = CustomData(
            gender=json_data.get('gender'),
            race_ethnicity=json_data.get('race_ethnicity'),
            parental_level_of_education=json_data.get('parental_level_of_education'),
            lunch=json_data.get('lunch'),
            test_preparation_course=json_data.get('test_preparation_course'),
            reading_score=float(json_data.get('reading_score')),
            writing_score=float(json_data.get('writing_score'))
        )
        
        # Convert to DataFrame
        input_df = data.get_data_as_dataframe()
        
        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        
        # Make prediction
        results = predict_pipeline.predict(input_df)
        
        return jsonify({
            'status': 'success',
            'prediction': float(results[0])
        })
    except Exception as e:
        logging.error(f"API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)