import os
os.environ["HF_HOME"] = "./hf_cache"
from transformers import BertTokenizer, BertModel
import torch

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def process_text(self, text):
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        print("\n--- Tokenized Output ---")
        print(inputs)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state

        print("\n--- Embedding Shape ---")
        print(embeddings.shape)

        return embeddings


# -------------------------------
# TEST RUN
# -------------------------------
if __name__ == "__main__":
    processor = TextPreprocessor()

    text = "A beautiful red flower in a garden"

    embedding = processor.process_text(text)

    print("\n--- Final Embedding ---")
    print(embedding)