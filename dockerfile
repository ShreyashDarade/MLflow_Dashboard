# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY mlflow_utils.py ,streamlit_app.py ,train_multiple_models.py ,train_random_forest.py /app/

# Expose port 8501 (default port for Streamlit)
EXPOSE 8501

# Default command: run the Streamlit dashboard.
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
