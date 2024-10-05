# Use an official Python runtime as a parent image
FROM python:3.10

# Set environment variables to prevent Python from writing pyc files to disc and buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install system libraries required for compiling Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev \    
    python3-dev \    
    libblas-dev \    
    liblapack-dev \
    # Additional dependencies can be added here
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Command to run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
