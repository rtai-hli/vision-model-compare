from huggingface_hub import login, whoami, HfApi
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def check_hf_access():
    # Login
    token = os.getenv('HUGGINGFACE_TOKEN')
    login(token=token)
    
    # Verify login
    user = whoami()
    print(f"Logged in as: {user['name']}")
    
    # List Meta's models
    api = HfApi()
    models = list(api.list_models(author="meta-llama"))
    print("\nAvailable Meta models:")
    for model in models:
        print(f"- {model.modelId}")

if __name__ == "__main__":
    check_hf_access()