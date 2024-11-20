from flask import Flask, request, render_template, jsonify
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
import base64
import io
import requests
from huggingface_hub import HfApi, get_token
from dotenv import load_dotenv
import os
import time
from datasets import load_dataset
import random

app = Flask(__name__)

# Get token from HuggingFace cache instead of .env
try:
    token = get_token()  # This gets the token from the cache
    if not token:
        raise ValueError("No HuggingFace token found in cache. Please run 'huggingface-cli login'")
    
    # Verify token
    api = HfApi()
    user_info = api.whoami()
    print(f"Successfully verified token for user: {user_info['name']}")
except Exception as e:
    print(f"Error verifying token: {str(e)}")
    raise

# Initialize Llama Vision with CPU offloading
try:
    llama_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    print(f"Loading model: {llama_model_id}")
    
    llama_processor = AutoProcessor.from_pretrained(
        llama_model_id,
        token=token,
        trust_remote_code=True
    )
    
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_id,
        token=token,
        device_map="auto",
        offload_folder="offload",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Add after Flask app initialization but before model loading
try:
    print("Loading ChartBench demo dataset...")
    demo_dataset = load_dataset("SincereX/ChartBench-Demo")
    print(f"Dataset loaded with {len(demo_dataset['train'])} examples")
except Exception as e:
    print(f"Error loading demo dataset: {str(e)}")
    demo_dataset = None

def generate_llama_response(image, processor, model):
    """Generate response from Llama 3.2 Vision model"""
    start_time = time.time()
    
    inputs = processor(images=image, text="Describe this image in detail", return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    elapsed_time = time.time() - start_time
    
    return response, elapsed_time

def generate_llava_response(image_base64):
    """Generate response from LLaVA model through Ollama"""
    start_time = time.time()
    api_url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "llava:13b",
        "prompt": "Describe this image in detail",
        "images": [image_base64],
        "stream": False
    }
    
    response = requests.post(api_url, json=payload)
    elapsed_time = time.time() - start_time
    
    if response.status_code == 200:
        return response.json()['response'], elapsed_time
    else:
        raise Exception(f"Ollama API error: {response.text}")

@app.route('/')
def home():
    return render_template('index.html', has_demos=(demo_dataset is not None))

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image_data = request.json['image']
        base64_image = image_data.split(',')[1] if ',' in image_data else image_data
        
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))
        
        llama_response, llama_time = generate_llama_response(image, llama_processor, llama_model)
        llava_response, llava_time = generate_llava_response(base64_image)
        
        return jsonify({
            'llama_response': llama_response,
            'llama_time': f"{llama_time:.2f}",
            'llava_response': llava_response,
            'llava_time': f"{llava_time:.2f}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo_images')
def get_demo_images():
    if demo_dataset is None:
        return jsonify({'error': 'Demo dataset not available'}), 500
    
    # Get 5 random examples from the dataset
    sample_indices = random.sample(range(len(demo_dataset['train'])), 5)
    samples = [demo_dataset['train'][i] for i in sample_indices]
    
    demo_images = []
    for sample in samples:
        demo_images.append({
            'image': sample['image'],
            'description': sample['text']
        })
    
    return jsonify({'demo_images': demo_images})

if __name__ == '__main__':
    app.run(debug=True)
