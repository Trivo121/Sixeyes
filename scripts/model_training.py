import os
import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# Paths
PROCESSED_DATA_PATH = os.path.normpath(r"D:\Study\Projects\Six_eyes\Spam_call_detection\Data\processed")
MODEL_SAVE_PATH = os.path.normpath(r"D:\Study\Projects\Six_eyes\Spam_call_detection\Models\nlp_model")

# Load processed metadata from Step-1
with open(os.path.join(PROCESSED_DATA_PATH, "processed_metadata.json"), "r") as f:
    data = json.load(f)

# Prepare the dataset for training
texts = []
labels = []

for entry in data:
    if "transcript" in entry and entry["transcript"]:  # Check if transcript exists and is non-empty
        texts.append(entry["transcript"])
        labels.append(1 if "fraud" in entry.get("file_name", "").lower() else 0)

# Ensure lengths match
assert len(texts) == len(labels), "Mismatch in lengths of texts and labels"

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize the data
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})
test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})
datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

# Fine-tune DistilBERT
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    eval_strategy="epoch",  # Updated parameter
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
    tokenizer=tokenizer,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
