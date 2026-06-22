# Start from a small official Python image
FROM python:3.11-slim

# Everything inside the container lives in /app
WORKDIR /app

# Install dependencies first (this layer is cached, so rebuilds are fast)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and the 3 data files it needs
COPY app_1.py .
COPY recommender_data.npz movies_meta.csv user_seen.json ./

# Streamlit serves on port 8501
EXPOSE 8501

# Start the app. --server.address=0.0.0.0 makes it reachable from outside the container.
CMD ["streamlit", "run", "app_1.py", "--server.port=8501", "--server.address=0.0.0.0"]
