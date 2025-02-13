import sys
import time
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.keyword_detection import get_detector
from utils.config import REALTIME_CONFIG
from utils.logger import setup_logger

logger = setup_logger("detection_logger")

class CallProcessor:
    def __init__(self):
        self.detector = get_detector()
        self.batch_buffer = []
        self.last_processed = time.time()
    
    def process_call(self, call_data: dict) -> dict:
        """Process a single call in real-time"""
        try:
            start_time = time.time()
            
            if time.time() - self.last_processed > REALTIME_CONFIG["processing_timeout"]:
                self._flush_buffer()
            
            result = self.detector._process_single(call_data["transcript"])
            
            logger.info(f"Processed call from {call_data['caller_id']} in {time.time() - start_time:.2f}s")
            
            return {
                **call_data,
                "prediction": result["is_fraud"],
                "confidence": result["confidence"],
                "keywords": result["keywords"],
                "decision": "block" if result["is_fraud"] else "allow"
            }
            
        except Exception as e:
            logger.error(f"Error processing call: {str(e)}")
            return {
                **call_data,
                "error": "Processing failed",
                "decision": "review"
            }

    def _flush_buffer(self):
        """Process buffered calls"""
        if self.batch_buffer:
            self.process_batch(self.batch_buffer)
            self.batch_buffer = []
        self.last_processed = time.time()

    def process_batch(self, call_batch: list) -> list:
        """Process multiple calls in batch mode"""
        try:
            start_time = time.time()
            results = self.detector.detect_batch([call["transcript"] for call in call_batch])
            
            for call, result in zip(call_batch, results):
                call.update(result)
                call["decision"] = "block" if result["is_fraud"] else "allow"
            
            logger.info(f"Processed {len(call_batch)} calls in {time.time() - start_time:.2f}s")
            return call_batch
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return [{"error": "Batch processing failed"} for _ in call_batch]

if __name__ == "__main__":
    # Example usage
    processor = CallProcessor()
    
    test_calls = [
        {
            "caller_id": "+1234567890",
            "transcript": "Your Social Security number has been suspended. Confirm your details immediately.",
            "duration": 45,
            "location": "New York"
        },
        {
            "caller_id": "+1987654321",
            "transcript": "Reminder about tomorrow's 2 PM meeting",
            "duration": 30,
            "location": "London"
        }
    ]
    
    print("Real-time processing:")
    for call in test_calls:
        result = processor.process_call(call)
        print(f"Call from {result['caller_id']}: {result['decision'].upper()} ({result.get('confidence', 0):.2f})")
    
    print("\nBatch processing:")
    batch_results = processor.process_batch(test_calls)
    for result in batch_results:
        status = "BLOCK" if result["decision"] == "block" else "ALLOW"
        print(f"{status}: {result['caller_id']} | Keywords: {result.get('keywords', [])}")