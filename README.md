# Fire and Smoke Detection API

This FastAPI application provides endpoints for detecting fire or smoke in images using two Vision Language Models (VLMs): SmolVLM and Moondream.

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
# Create a virtual environment with Python 3.10
python3.10 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# Run the FastAPI application
python app.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

The application provides two endpoints:

1. `/predict/smolvlm` - Uses SmolVLM model for prediction
2. `/predict/moondream` - Uses Moondream model for prediction

You can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`

## System Requirements

- Python 3.10
- Sufficient disk space for model downloads
- For GPU acceleration: Compatible GPU with appropriate drivers

## Note

The first run might take some time as it needs to download the model weights from HuggingFace. 