# Step 1: Use Python 3.10 (stable for ML libraries)
FROM python:3.10-slim

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Install system dependencies (needed for numpy, pandas, sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Step 4: Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all project files into container
COPY . .

# Step 6: Expose Render’s port
EXPOSE 10000

# Step 7: Run the app with Gunicorn
CMD ["gunicorn", "app:app", "--workers=1", "--threads=4", "--timeout=120", "--bind", "0.0.0.0:$PORT"]
