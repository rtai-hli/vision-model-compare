# Vision Model Comparison

A web interface to compare different vision-language models using Ollama. Currently compares:
- LLaVA (Latest)
- LLaVA 13B

## Setup

1. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install Ollama and required models:
```bash
# Install Ollama from https://ollama.ai
ollama pull llava
ollama pull llava:13b
```

3. Run the application:
```bash
python app.py
```

4. Open http://127.0.0.1:5000 in your browser

## Features

- Upload custom images or use demo charts from ChartBench dataset
- Compare responses from different vision models
- View processing time for each model
- Interactive UI with tooltips and visual feedback

## Requirements

- Python 3.10+
- Ollama installed and running
- 8GB+ RAM recommended
