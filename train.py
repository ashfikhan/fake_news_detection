import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Optional GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the data
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

df_fake['label'] = 0
df_real['label'] = 1

# Combine and sample a smaller dataset for faster training
df = pd.concat([df_fake, df_real]).sample(n=3000, random_state=42).reset_index(drop=True)
df['text'] = df['title'] + " " + df['text']
df = df[['text', 'label']]

# Split dataset
train_texts, val_texts = train_test_split(df, test_size=0.1, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=512)

train_dataset = Dataset.from_pandas(train_texts).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_texts).map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["text", "__index_level_0__"])
val_dataset = val_dataset.remove_columns(["text", "__index_level_0__"])
train_dataset.set_format("torch")
val_dataset.set_format("torch")

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# Training arguments (faster config)
training_args = TrainingArguments(
    output_dir="bert_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_dir="logs",
    logging_steps=10,
    load_best_model_at_end=True,
    disable_tqdm=False
)

print("Training samples:", len(train_dataset)) 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Start training
trainer.train()

# Save model and tokenizer manually (safe fallback)
trainer.save_model("bert_model")
tokenizer.save_pretrained("bert_model")

# Save manually in case Trainer didn’t save it right
model_to_save = trainer.model
torch.save(model_to_save.state_dict(), "bert_model/pytorch_model.bin")
model_to_save.config.to_json_file("bert_model/config.json")

print("✅ Training complete. Model and tokenizer saved.")

