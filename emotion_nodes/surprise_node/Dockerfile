# Base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn transformers
RUN pip install torch

# Copy local files to the container
COPY surprise_node.py .
COPY surprise_classification_model.pth .
COPY surprise_tokenizer /app/tokenizer



# Expose port 8000
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "surprise_node:app", "--host", "0.0.0.0", "--port", "8000"]
