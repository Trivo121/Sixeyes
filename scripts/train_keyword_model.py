from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np
from utils.config import MODEL_PATH, TRAINING_CONFIG, BASE_DIR

# Load metrics
metric = load_metric("precision_recall_fscore_support")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = metric.compute(
        predictions=predictions,
        references=labels,
        average="weighted"
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def load_custom_dataset():
    # Load preprocessed CSV data
    df = pd.read_csv(BASE_DIR / "data/processed/cleaned_calls.csv")
    
    # Convert to Hugging Face dataset format
    dataset = Dataset.from_pandas(df)
    
    # Rename columns if necessary (adapt to your CSV structure)
    dataset = dataset.rename_column("is_spam", "label")  # Change "is_spam" to your label column name
    dataset = dataset.rename_column("transcript", "text")  # Change "transcript" to your text column name
    
    return dataset.train_test_split(test_size=0.2)

def train_model():
    # Load custom dataset
    dataset = load_custom_dataset()
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=TRAINING_CONFIG["max_seq_length"]
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_PATH),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=TRAINING_CONFIG["learning_rate"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["batch_size"],
        num_train_epochs=TRAINING_CONFIG["epochs"],
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(BASE_DIR / "logs")
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )
    
    # Start training
    train_result = trainer.train()
    trainer.save_model()
    
    # Save training metrics
    trainer.save_metrics("train", train_result.metrics)
    
    # Evaluate and print final metrics
    eval_metrics = trainer.evaluate()
    print("\nTraining complete. Final metrics:")
    print(f"Precision: {eval_metrics['eval_precision']:.4f}")
    print(f"Recall: {eval_metrics['eval_recall']:.4f}")
    print(f"F1 Score: {eval_metrics['eval_f1']:.4f}")

if __name__ == "__main__":
    train_model() 