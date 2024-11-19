from flask import Flask, request, render_template, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import base64
import io

app = Flask(__name__)

# Initialize models and processors
model1_name = "Salesforce/blip-image-captioning-base"
model2_name = "Salesforce/blip-image-captioning-large"

processor1 = BlipProcessor.from_pretrained(model1_name)
model1 = BlipForConditionalGeneration.from_pretrained(model1_name, torch_dtype=torch.float16)

processor2 = BlipProcessor.from_pretrained(model2_name)
model2 = BlipForConditionalGeneration.from_pretrained(model2_name, torch_dtype=torch.float16)

# Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model1 = model1.to(device)
model2 = model2.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get the image from the request
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process image with both models
        inputs1 = processor1(images=image, return_tensors="pt").to(device)
        inputs2 = processor2(images=image, return_tensors="pt").to(device)
        
        # Generate responses
        with torch.no_grad():
            outputs1 = model1.generate(
                **inputs1,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
            
            outputs2 = model2.generate(
                **inputs2,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        # Decode responses
        response1 = processor1.decode(outputs1[0], skip_special_tokens=True)
        response2 = processor2.decode(outputs2[0], skip_special_tokens=True)
        
        return jsonify({
            'model1_name': model1_name,
            'model2_name': model2_name,
            'response1': response1,
            'response2': response2
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
