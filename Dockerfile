# Use a lightweight official Python 3.12 runtime to match your local system
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install updated modern system dependencies required for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container first to leverage Docker caching
COPY requirements.txt .

# Install the Python dependencies listed in requirements.txt (Ensure tensorflow==2.16.1 is inside)
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files needed to run the Flask web service
COPY app.py best_model.h5 labels.txt ./

# Expose port 5000 (the standard Flask port) to the outside world
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]