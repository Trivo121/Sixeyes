import os
import json
import librosa
import whisper
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Define paths
RAW_DATA_PATH = os.path.normpath(r"D:\Study\Projects\Six_eyes\Spam_call_detection\Data\raw")
PROCESSED_DATA_PATH = os.path.normpath(r"D:\Study\Projects\Six_eyes\Spam_call_detection\Data\processed")
TRANSCRIPTS_PATH = os.path.normpath(r"D:\Study\Projects\Six_eyes\Spam_call_detection\Data\transcripts")
METADATA_FILE = os.path.normpath(r"D:\Study\Projects\Six_eyes\Spam_call_detection\Data\raw\metadata.csv")

# Ensure processed directories exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(TRANSCRIPTS_PATH, exist_ok=True)

# Text preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer() 

# Initialize Whisper model
whisper_model = whisper.load_model("base")  # Options: "tiny", "base", "small", "medium", "large"

def clean_text(text):
    """Clean and preprocess the text transcript."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return " ".join(words)

def extract_audio_features(file_path):
    """Extract audio features using librosa."""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist()
        return {"duration": duration, "mfcc": mfcc}
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None

def transcribe_audio(file_path):
    """Transcribe audio to text using Whisper."""
    try:
        result = whisper_model.transcribe(file_path)
        transcript = result["text"]
        return clean_text(transcript)
    except Exception as e:
        logger.error(f"Error transcribing file {file_path}: {e}")
        return None

def process_audio_file(file_path):
    """Process a single audio file (extract features, transcribe, and save)."""
    try:
        # Extract audio features
        features = extract_audio_features(file_path)

        # Transcribe audio
        transcript = transcribe_audio(file_path)
        if transcript:
            # Save transcript to file
            transcript_file = os.path.join(TRANSCRIPTS_PATH, f"{os.path.splitext(os.path.basename(file_path))[0]}.txt")
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(transcript)

        return {
            "file_name": os.path.basename(file_path),
            "features": features,
            "transcript": transcript,
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None

def preprocess_pipeline():
    """Preprocess raw data and save processed metadata."""
    processed_metadata = []

    # Load and preprocess metadata
    if os.path.exists(METADATA_FILE):
        try:
            metadata = pd.read_csv(METADATA_FILE)
            logger.info("Metadata loaded successfully!")
            # Add preprocessing steps for metadata if necessary
        except Exception as e:
            logger.error(f"Error loading metadata file: {e}")
            metadata = None
    else:
        logger.warning("Metadata file not found. Proceeding without metadata.")

    audio_dir = os.path.join(RAW_DATA_PATH, "audio-wav-16khz")
    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(audio_dir)
        for file in files if file.endswith(".wav")
    ]

    # Process audio files with threading for speed
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_audio_file, audio_files), total=len(audio_files), desc="Processing audio files"))

    # Filter out None results and save metadata
    processed_metadata = [res for res in results if res is not None]
    with open(os.path.join(PROCESSED_DATA_PATH, "processed_metadata.json"), "w") as f:
        json.dump(processed_metadata, f, indent=4)

    logger.info("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_pipeline()
