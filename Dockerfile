# Use the official Selenium Standalone Chrome image as the base image
FROM python:slim

# Switch to root user for installation
USER root

# Install necessary packages and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container and install dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your FastAPI Python script to the container
COPY . .

# Set the command to run your Python script
CMD ["python3", "main.py"]
