from transformers import BertForSequenceClassification, BertTokenizer
import os

# Print the current working directory to be sure where files will be saved.
print("Current working directory:", os.getcwd())

# Load the base model and tokenizer without any training modifications.
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Try saving the model and tokenizer manually.
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

print("✅ Save attempted. Check the 'bert_model' folder.")
