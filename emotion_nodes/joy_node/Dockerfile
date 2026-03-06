# Base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy local files to the container
COPY joy_node.py .
COPY joy_classification_model.pth .
COPY joy_tokenizer /app/tokenizer

# Install dependencies
RUN pip install fastapi uvicorn transformers torch

# Expose port 8000
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "joy_node:app", "--host", "0.0.0.0", "--port", "8000"]
