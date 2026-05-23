import requests

# The URL where your Docker container is running Flask
url = "http://localhost:5000/predict"

# Path to any sample image you have in your folder (e.g., a building, forest, street, etc.)
# Replace 'test_image.jpg' with an actual image file name in your folder!
image_path = "test_image.jpg" 

try:
    print("Sending image to Docker container...")
    with open(image_path, "rb") as img:
        files = {"file": img}
        response = requests.post(url, files=files)
    
    # Print the result from the model
    if response.status_code == 200:
        print("\n--- Prediction Success! ---")
        print(response.json())
    else:
        print(f"\nError {response.status_code}: {response.text}")

except FileNotFoundError:
    print(f"Error: Could not find the file '{image_path}'. Make sure it's in this folder!")
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the API. Is your Docker container running?")