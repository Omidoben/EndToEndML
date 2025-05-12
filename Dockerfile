# Base image: Uses a lightweight version of Python 3.8. - "Slim" means it's minimal (faster builds, smaller image)
FROM python:3.13-slim           

# Sets the working directory inside the container to /app
WORKDIR /app

# Copies requirements.txt from your host machine to /app in the container - Installs Python dependencies without caching, to reduce image size.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies everything from your project directory on the host into /app in the container
COPY . .

# ENV PYTHONPATH="/app"

# Creates an artifacts directory inside the container — maybe used to store trained models, logs, etc
RUN mkdir -p artifacts

# Runs the training pipeline script during build time. This creates and saves your trained model and transformer artifacts inside the containe
# RUN python -m src.pipeline.train_pipeline

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

# Uses Gunicorn (a production WSGI server) to serve your Flask app
# --bind 0.0.0.0:8080: Listen on all interfaces at port 8080.
# app:app: Tells Gunicorn to use the Flask app object called app in app.py