# src/chatbot.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH

class Chatbot:
    def __init__(self):
        # Load the model and tokenizer from Hugging Face model hub
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    def generate_response(self, prompt: str) -> str:
        """
        Generates a chatbot response for a given prompt.

        Parameters:
        - prompt (str): The user's input prompt.

        Returns:
        - str: The generated response.
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs,
            max_length=MAX_LENGTH,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=75,
            top_p=0.95,
            temperature=0.7,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
