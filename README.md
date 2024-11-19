# Llama Vision Web App

A Flask-based web application that uses Llama 3.2 Vision model for image analysis.

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Click the "Choose Image" button to select an image from your computer
2. The image will be displayed and automatically sent to the Llama Vision model for analysis
3. The model's response will appear below the image

## Requirements

- Python 3.8+
- Sufficient RAM for running Llama Vision model (recommended: 16GB+)
- CUDA-capable GPU (recommended)
