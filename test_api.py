import requests
import json

# Change this to your local or deployed API URL
API_URL = "http://localhost:12000"  # Using the port from RUNTIME_INFORMATION

def test_root():
    response = requests.get(f"{API_URL}/")
    print("Root endpoint response:", response.json())
    
def test_socialize_context():
    data = {
        "context": "Alice and Bob are at a coffee shop. Alice is a student and Bob is a professor. They are discussing a research project.",
        "task_specific_instructions": "Focus on the academic relationship dynamics."
    }
    response = requests.post(f"{API_URL}/socialize-context", json=data)
    print("Socialize context response:", json.dumps(response.json(), indent=2))
    return response.json()

if __name__ == "__main__":
    print("Testing API endpoints...")
    test_root()
    test_socialize_context()