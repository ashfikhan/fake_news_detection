from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "./bert_model"  # Use the correct relative or absolute path
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# Define prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    return "Real" if predicted_class == 1 else "Fake"

# Example usage
sample_text = "Breaking news: This just happened in the world of politics..."
print("Prediction:", predict(sample_text))
