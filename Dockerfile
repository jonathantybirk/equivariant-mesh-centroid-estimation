# Use a lightweight Python 3.13 image
FROM python:3.13-slim

# Install system dependencies required by some packages and for visualization (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the entire repository into the container
COPY . /app

# Upgrade pip and install project dependencies in editable mode
RUN pip install --upgrade pip
RUN pip install -e .

# Expose a port if needed (e.g., for TensorBoard or similar)
EXPOSE 6006

# Default command: run the training script
CMD ["python", "-m", "src.train.train_model"]
