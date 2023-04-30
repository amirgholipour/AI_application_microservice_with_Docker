# # Use the official lightweight Python image as the base image
# FROM python:3.9-slim
# FROM ubuntu:latest
FROM python:3.9.16-slim-buster
# Install the required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git-all
# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --upgrade pip
RUN pip install github
RUN pip install --upgrade --force-reinstall torch torchvision
RUN pip install git+https://github.com/huggingface/transformers
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip3 install opencv-python-headless==4.5.3.56

# Copy the rest of your application code into the container
# COPY . .
COPY data /app
COPY ImageNetLabels.txt /app/ImageNetLabels.txt
COPY yolov8l.pt /app/yolov8l.pt
COPY model.py /app/model.py
#COPY inference.py /app/inference.py
COPY app.py /app/app.py
RUN rm -rf /tmp/*
# Expose the port the app runs on
EXPOSE 5000

# Start the application
CMD ["python", "app.py"]
