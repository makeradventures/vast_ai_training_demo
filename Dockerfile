FROM pytorch/pytorch:latest

WORKDIR /workspace

# Install additional dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy training code
COPY train.py .

# Create directories
RUN mkdir -p checkpoints data

CMD ["python", "train.py"]
