from flask import Flask, request, render_template, jsonify
from PIL import Image
import base64
import io
import requests
import time
from datasets import load_dataset
import random

app = Flask(__name__)

# Load demo dataset
try:
    print("Loading ChartBench demo dataset...")
    demo_dataset = load_dataset("SincereX/ChartBench-Demo")
    print(f"Dataset loaded with {len(demo_dataset['train'])} examples")
except Exception as e:
    print(f"Error loading demo dataset: {str(e)}")
    demo_dataset = None

def generate_ollama_response(model_name, image_base64):
    """Generate response from Ollama model"""
    start_time = time.time()
    api_url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": "Describe this image in detail",
        "images": [image_base64],
        "stream": False
    }
    
    try:
        print(f"Sending request to Ollama for model: {model_name}")
        response = requests.post(api_url, json=payload)
        print(f"Ollama response status: {response.status_code}")
        print(f"Ollama response content: {response.text[:200]}...")  # Print first 200 chars
        
        response.raise_for_status()
        elapsed_time = time.time() - start_time
        return response.json()['response'], elapsed_time
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API for {model_name}: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html', has_demos=(demo_dataset is not None))

@app.route('/demo_images')
def get_demo_images():
    if demo_dataset is None:
        print("Demo dataset is not available")
        return jsonify({'demo_images': []}), 200
    
    try:
        # Get 5 random examples from the dataset
        sample_indices = random.sample(range(len(demo_dataset['train'])), 5)
        samples = [demo_dataset['train'][i] for i in sample_indices]
        
        demo_images = []
        for sample in samples:
            print(f"Sample keys: {sample.keys()}")
            # Convert PIL Image to base64
            image = sample['image']
            if isinstance(image, Image.Image):
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
            else:
                img_str = ''
                
            demo_images.append({
                'image': img_str,
                'description': 'Chart example'  # Default description since dataset doesn't have captions
            })
        
        print(f"Returning {len(demo_images)} demo images")
        return jsonify({'demo_images': demo_images})
    except Exception as e:
        print(f"Error processing demo images: {str(e)}")
        return jsonify({'demo_images': []}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("Received analyze request")
        image_data = request.json['image']
        base64_image = image_data.split(',')[1] if ',' in image_data else image_data
        
        print("Processing with LLaVA models...")
        try:
            latest_response, latest_time = generate_ollama_response("llava", base64_image)
            print("LLaVA latest completed")
            
            v13b_response, v13b_time = generate_ollama_response("llava:13b", base64_image)
            print("LLaVA 13B completed")
            
            result = {
                'latest_response': latest_response,
                'latest_time': f"{latest_time:.2f}",
                'v13b_response': v13b_response,
                'v13b_time': f"{v13b_time:.2f}"
            }
            print(f"Returning result: {str(result)[:200]}...")  # Print first 200 chars
            return jsonify(result)
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return jsonify({'error': f'Error generating response: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Error in analyze route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
