import requests
import json


BASE_URL = "http://localhost:8000"


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    print(f"Response: {response.json()}")


def test_single_prediction():
    test_cases = [
        "Looking forward to the demo!",
        "Not interested please remove",
        "Can we discuss pricing",
        "This is amazing let's proceed",
        "Bad service very disappointed"
    ]

    for text in test_cases:
        payload = {"text": text}
        response = requests.post(f"{BASE_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            print(f"Text: '{text}'")
            print(f"Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
            print(f"Probabilities: {result['probabilities']}")
            print("-" * 50)
        else:
            print(f"Error for '{text}': {response.status_code} - {response.text}")


def test_batch_prediction():
    texts = [
        "Can you send me pricing details",
        "Not sure about this",
        "Terrible experience"
    ]

    response = requests.post(f"{BASE_URL}/predict_batch", json=texts)
    print(f"Batch prediction: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("Testing Sentiment Analysis API...")
    test_health()
    print("\n" + "=" * 60 + "\n")
    test_single_prediction()
    print("\n" + "=" * 60 + "\n")
    test_batch_prediction()