import requests
import json

def test_ollama_connection():
    """Test connection to local Ollama server with basic prompt"""
    
    # Set headers with User-Agent and update URL to use v1 endpoint
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0)',
        'Content-Type': 'application/json'
    }
    
    # First check if Ollama service is running
    try:
        health_check = requests.get("http://127.0.0.1:11434", 
                                  headers=headers,
                                  timeout=5)
        health_check.raise_for_status()
        print("Ollama service is running")
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to Ollama service")
        print("Please ensure Ollama is installed and running on port 11434")
        print("Try running 'OLLAMA_HOST=0.0.0.0:11434 ollama serve' in a terminal")
        return
    except requests.exceptions.HTTPError as e:
        print(f"\nError: Ollama service returned an error - {str(e)}")
        print("Please check if Ollama is properly configured")
        return
    except Exception as e:
        print(f"\nUnexpected error connecting to Ollama: {str(e)}")
        print("Please ensure Ollama is installed and running correctly")
        return

    # Test generate endpoint
    try:
        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "llama3.2:latest",
            "prompt": "hello are you there",
            "stream": False  # Set to False to get a single response
        }
        
        print("\nSending test prompt to Ollama...")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Process the response line by line
        full_response = ""
        for line in response.text.splitlines():
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        full_response += data['response']
                except json.JSONDecodeError:
                    continue
        
        print("\nOllama response:")
        print(full_response if full_response else "No response received")
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Ollama: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_ollama_connection()
