# Use an official Python runtime as a parent image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./ /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y ffmpeg portaudio19-dev
RUN pip install pyaudio


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt