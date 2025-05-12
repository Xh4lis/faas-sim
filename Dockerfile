FROM python:3.7-slim

WORKDIR /app

# Install dependencies required for visualization
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt
RUN pip install jupyter matplotlib seaborn

# Copy project files
COPY . .
RUN pip install -e .

# Create output directory for results
RUN mkdir -p /output

# Expose port for Jupyter
EXPOSE 8888

# Default command runs Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.notebook_dir='/app'"]