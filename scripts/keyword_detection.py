import torch
import re
from transformers import pipeline
from typing import List, Dict
from utils.file_handler import load_model_and_tokenizer, load_detection_patterns
from utils.config import REALTIME_CONFIG

class FraudKeywordDetector:
    def __init__(self):
        self.model, self.tokenizer = load_model_and_tokenizer()
        self.patterns = load_detection_patterns()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def _preprocess_batch(self, texts: List[str]) -> List[str]:
        return [text.lower().strip() for text in texts]

    def _detect_keywords(self, text: str) -> List[str]:
        detected = []
        
        # Match exact keywords with word boundaries
        for keyword in self.patterns["keywords"]:
            if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE):
                detected.append(keyword)
        
        # Match regex patterns
        for pattern in self.patterns["regex_patterns"]:
            if pattern.search(text):
                detected.append(pattern.pattern)
        
        return list(set(detected))

    def _process_single(self, text: str) -> Dict:
        try:
            processed_text = self._preprocess_batch([text])[0]
            prediction = self.classifier(
                processed_text,
                truncation=True,
                max_length=512
            )[0]
            
            keywords = self._detect_keywords(processed_text)
            
            return {
                "text": text,
                "is_fraud": prediction["label"] == "LABEL_1",
                "confidence": prediction["score"],
                "keywords": keywords,
                "pattern_matches": len(keywords) > 0
            }
        except Exception as e:
            return {
                "text": text,
                "error": str(e),
                "is_fraud": False
            }

    def detect_batch(self, texts: List[str]) -> List[Dict]:
        processed_texts = self._preprocess_batch(texts)
        predictions = self.classifier(
            processed_texts,
            truncation=True,
            padding=True,
            max_length=512,
            batch_size=REALTIME_CONFIG["max_batch_size"]
        )
        
        results = []
        for text, pred in zip(texts, predictions):
            processed_text = self._preprocess_batch([text])[0]
            keywords = self._detect_keywords(processed_text)
            
            results.append({
                "text": text,
                "is_fraud": pred["label"] == "LABEL_1",
                "confidence": pred["score"],
                "keywords": keywords,
                "pattern_matches": len(keywords) > 0,
                "decision": (
                    "block" if pred["score"] > REALTIME_CONFIG["confidence_threshold"] 
                    else "review"
                )
            })
        
        return results

def get_detector() -> FraudKeywordDetector:
    return FraudKeywordDetector()