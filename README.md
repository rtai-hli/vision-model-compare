# Vision Model Comparison

A web interface to compare different vision-language models using Ollama. Currently compares:
- LLaVA (Latest)
- LLaVA 13B

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Click the "Choose Image" button to select an image from your computer
2. The image will be displayed and automatically sent to the Llama Vision model for analysis
3. The model's response will appear below the image

## Requirements

- Python 3.8+
- Sufficient RAM for running Llama Vision model (recommended: 16GB+)
- CUDA-capable GPU (recommended)
