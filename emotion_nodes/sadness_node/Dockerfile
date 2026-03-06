# Base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy local files to the container
COPY sadness_node.py .
COPY sadness_classification_model.pth .
COPY sadness_tokenizer /app/tokenizer

# Install dependencies
RUN pip install fastapi uvicorn transformers torch

# Expose port 8000
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "sadness_node:app", "--host", "0.0.0.0", "--port", "8000"]
