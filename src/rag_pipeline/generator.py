# src/rag_pipeline/generator.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Generator:
    """
    Generator uses a language model to produce an answer conditioned on retrieved context.

    Attributes:
        model_name (str): Name or path of the generation model.
        model (PreTrainedModel): The loaded Hugging Face model.
        tokenizer (PreTrainedTokenizer): The tokenizer for the model.
    """
    def __init__(self, model_name: str):
        """
        Args:
            model_name (str): Name/path of the Hugging Face generation model.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()

    def generate_answer(self, context: str, query: str, max_length: int = 200):
        """
        Generate an answer to the query using the given context.

        Args:
            context (str): The retrieved documents context.
            query (str): The user's query.
            max_length (int): Max length of the generated text.

        Returns:
            str: The generated answer.
        """
        # Prompt format (simple): "Context: ... Query: ..."
        prompt = f"Context: {context}\nQuery: {query}\nAnswer:"
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=max_length, do_sample=True, top_p=0.9, temperature=0.8)
        answer = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return answer.strip()
