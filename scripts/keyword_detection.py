from transformers import pipeline
import json

# Path to processed metadata file
PROCESSED_DATA_PATH = "D:/Study/Projects/Six_eyes/Spam_call_detection/Data/processed"
METADATA_FILE = "processed_metadata.json"

# Load the data
with open(f"{PROCESSED_DATA_PATH}/{METADATA_FILE}", "r") as file:
    data = json.load(file)


# Load the fine-tuned model
MODEL_SAVE_PATH = "D:/Study/Projects/Six_eyes/Spam_call_detection/Models/nlp_model"

MODEL_PATH = MODEL_SAVE_PATH
nlp_model = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)

def detect_fraud_keywords(transcript):
    """Detect fraudulent content in the transcript."""
    prediction = nlp_model(transcript)
    label = prediction[0]["label"]
    confidence = prediction[0]["score"]

    if label == "LABEL_1":  # LABEL_1 corresponds to "fraudulent"
        print(f"Fraudulent content detected with confidence: {confidence:.2f}")
        return True
    return False

# Example usage
for entry in data:
    transcript = entry["transcript"]
    if detect_fraud_keywords(transcript):
        print(f"Potential fraud detected in file: {entry['file_name']}")
