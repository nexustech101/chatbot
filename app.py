# app.py

from flask import Flask, request, jsonify
from src.chatbot import Chatbot

# Initialize Flask app and chatbot instance
app = Flask(__name__)
chatbot = Chatbot()


@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint to receive a prompt and return a chatbot response.
    
    Request JSON:
    - prompt (str): The user's input prompt.

    Response JSON:
    - response (str): The generated response from the chatbot.
    """
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Generate response using chatbot
    response = chatbot.generate_response(prompt)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
