# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# copy the current directory contents into the container at /app
COPY . /app 

# Install dependencies
RUN pip install -r requirements.txt

# Make port 5000 available to the worldoutside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
