# Use Python 3.10 as base image
FROM python:3.10.9

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
