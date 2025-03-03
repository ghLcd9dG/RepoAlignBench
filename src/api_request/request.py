import requests

PROXY_URL = "http://192.168.211.164:5000/api/openai"

data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a Python function to reverse a string."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
}

try:
    response = requests.post(PROXY_URL, json=data)

    if response.status_code == 200:
        print("Response from OpenAI Proxy:")
        print(response.json()["choices"][0]["message"]["content"])
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

except Exception as e:
    print(f"Request failed: {e}")
