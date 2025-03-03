import requests
from flask import Flask, request, jsonify


app = Flask(__name__)

OPENAI_API_KEY = ""
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


@app.route("/api/openai", methods=["POST"])
def forward_to_openai():
    try:
        client_request_data = request.get_json()

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            OPENAI_API_URL, headers=headers, json=client_request_data
        )

        response_data = response.json()
        return jsonify(response_data), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
