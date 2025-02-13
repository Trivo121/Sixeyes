import pandas as pd
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils.file_handler import load_config
from utils.logger import setup_logger

logger = setup_logger('train_nlp_model')
config = load_config()

class SpamDataset(Dataset):
    """Dataset class for spam call transcripts"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model():
    # Load and prepare data
    df = pd.read_csv(config['PROCESSED_DATA_PATH'])
    texts = df['transcript'].fillna('').tolist()
    labels = df['is_spam'].astype(int).tolist()

    # Train-validation split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Tokenize datasets
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    # Create datasets
    train_dataset = SpamDataset(train_encodings, train_labels)
    val_dataset = SpamDataset(val_encodings, val_labels)

    # Model configuration
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['MODEL_PATH'],
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train and save model
    trainer.train()
    model.save_pretrained(config['MODEL_PATH'])
    tokenizer.save_pretrained(config['MODEL_PATH'])
    logger.info(f"Model saved to {config['MODEL_PATH']}")

if __name__ == '__main__':
    train_model()