---

# Chatbot Project

This project is a Python-based chatbot that uses a transformer model from Hugging Face’s `transformers` library, combined with a Flask API to serve responses. The chatbot is capable of generating conversational responses based on user input prompts and can be easily integrated with a front-end interface.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Customization](#customization)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project is intended to create a chatbot API that can be used to power conversational applications. It leverages Hugging Face’s transformer models, particularly the conversational model `DialoGPT`, for generating responses to user inputs. The chatbot is wrapped in a Flask API, allowing it to be accessed as a RESTful web service.

## Features

- **Transformer-based Model**: Utilizes state-of-the-art natural language processing models to generate human-like responses.
- **Flask API**: Provides an HTTP API endpoint to accept prompts and return chatbot responses, making it easy to integrate with web or mobile front-ends.
- **Configurable Model and Parameters**: Easily switch models and adjust parameters like response length and generation settings.
- **Open-Source**: The project is built on open-source libraries and is free to modify and extend.

---

## Project Structure

```plaintext
chatbot_project/
│
├── src/
│   ├── __init__.py        # Makes src a package
│   ├── chatbot.py         # Contains the chatbot logic and response generation
│   └── config.py          # Stores configuration for the model and tokenizer
│
├── app.py                 # Flask API to handle requests
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

### File Descriptions

- **`src/config.py`**: Configuration file for setting model parameters like model name and maximum response length.
- **`src/chatbot.py`**: Contains the `Chatbot` class, which loads the model and generates responses based on prompts.
- **`app.py`**: The main Flask application file, which sets up an API endpoint to receive prompts and return responses.
- **`requirements.txt`**: Lists all dependencies needed to run the project.

---

## Setup and Installation

### Prerequisites

- Python 3.7+
- `pip` (Python package installer)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/chatbot_project.git
   cd chatbot_project
   ```

2. **Install dependencies**:
   Use `pip` to install the required libraries.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application**:
   ```bash
   python app.py
   ```

   The Flask app will start on `http://127.0.0.1:5000/` by default.

---

## Usage

This project exposes a REST API that can be queried to generate chatbot responses. Once the server is running, you can send a `POST` request to the `/chat` endpoint with a JSON payload containing your input prompt.

### Example Usage

You can test the API using `curl` or any API client (e.g., Postman).

#### Using `curl`
```bash
curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you?"}'
```

#### Expected Response
```json
{
  "response": "I'm doing well! How can I help you today?"
}
```

---

## API Reference

### `POST /chat`

- **Description**: Accepts a user input prompt and returns a chatbot response generated by the transformer model.
- **Request Body**: JSON with the key `"prompt"` (string), representing the user input.
  ```json
  {
    "prompt": "Hello, how are you?"
  }
  ```
- **Response**: JSON with the key `"response"` containing the generated response.

  ```json
  {
    "response": "I'm good! How can I assist you today?"
  }
  ```

- **Error Handling**: If no prompt is provided, a `400` status code and an error message will be returned.

---

## Configuration

The configuration for the model and generation parameters is managed in `src/config.py`. Key parameters include:

- **`MODEL_NAME`**: Defines the model used for generating responses (default: `microsoft/DialoGPT-small`).
- **`MAX_LENGTH`**: The maximum length of the response generated (default: `50` tokens).

```python
# src/config.py

MODEL_NAME = "microsoft/DialoGPT-small"
MAX_LENGTH = 50
```

---

## Customization

You can customize the chatbot’s behavior by modifying parameters in `src/config.py` or using different transformer models available on the [Hugging Face Model Hub](https://huggingface.co/models).

- **Model Choice**: Change `MODEL_NAME` to any conversational model available on Hugging Face.
- **Generation Parameters**: Modify parameters like `MAX_LENGTH`, `temperature`, `top_k`, and `top_p` in `src/chatbot.py` to fine-tune response generation.

Example of adjusting generation parameters in `src/chatbot.py`:
```python
outputs = self.model.generate(
    inputs,
    max_length=MAX_LENGTH,
    temperature=0.7,      # Adjusts creativity in response
    top_k=50,             # Limits sampling pool for diversity
    top_p=0.95,           # Probability mass for nucleus sampling
)
```

---

## Examples

### Customizing Prompts
You can experiment with different types of prompts to see how the chatbot responds. For example:

```json
{
  "prompt": "What is your favorite programming language?"
}
```

**Expected Response**:
```json
{
  "response": "I'm a chatbot, so I don't code, but I hear Python is popular!"
}
```

### Changing Models
To use a different model (e.g., `gpt2`), modify the `MODEL_NAME` in `src/config.py`:
```python
MODEL_NAME = "gpt2"
```

---

## Dependencies

All dependencies are listed in `requirements.txt`:
- `transformers`: Library for transformer-based models.
- `torch`: PyTorch, used as the backend for model computation.
- `flask`: For building and serving the API.

To install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Contributing

If you'd like to contribute to this project:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

## Acknowledgments

- Hugging Face for providing the `transformers` library and hosting numerous pretrained models.
- The open-source community for contributing to libraries like Flask and PyTorch.

---
