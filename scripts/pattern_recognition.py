import re
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CallPatternAnalyzer:
    def __init__(self, user_contacts, time_window=24, similarity_threshold=0.75):
        self.user_contacts = user_contacts  # Known contact numbers
        self.time_window = timedelta(hours=time_window)  # Time window for pattern detection
        self.similarity_threshold = similarity_threshold  # Similarity threshold for repetitive lines
        
        self.call_history = defaultdict(list)  # Call history storage
        self.vectorizer = TfidfVectorizer(stop_words='english')  # TF-IDF for text similarity

    def is_unknown_number(self, caller_number):
        """Check if the caller is not in the user's contacts"""
        return caller_number not in self.user_contacts

    def detect_immediate_response(self, response_time):
        """Detect if the caller starts speaking immediately after call pickup"""
        IMMEDIATE_RESPONSE_THRESHOLD = 1.0  # 1 second threshold
        return response_time <= IMMEDIATE_RESPONSE_THRESHOLD

    def detect_repetitive_lines(self, current_call):
        """Detect repeated phrases in call history from the same number"""
        recent_calls = [
            call for call in self.call_history[current_call['caller_number']]
            if current_call['timestamp'] - call['timestamp'] <= self.time_window
        ]
        
        if not recent_calls:
            return False

        # Get past transcripts + current transcript
        transcripts = [call['transcript'] for call in recent_calls] + [current_call['transcript']]
        
        # Compute TF-IDF similarity
        tfidf_matrix = self.vectorizer.fit_transform(transcripts)
        similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        
        # Check if any past call is highly similar to current call
        return any(similarity >= self.similarity_threshold for similarity in similarity_matrix[0])

    def analyze_call_patterns(self, call_data):
        """Analyze call for spam patterns"""
        patterns = {}
        
        if self.is_unknown_number(call_data['caller_number']):
            patterns['unknown_number'] = True
            patterns['immediate_response'] = self.detect_immediate_response(call_data['response_time'])
        
        patterns['repetitive_content'] = self.detect_repetitive_lines(call_data)
        
        self.call_history[call_data['caller_number']].append(call_data)
        
        spam_confidence = self.calculate_confidence(patterns)
        
        logging.info(f"Analysis for {call_data['caller_number']}: {patterns} | Spam Confidence: {spam_confidence}")
        
        return {'patterns': patterns, 'spam_confidence': spam_confidence}

    def calculate_confidence(self, patterns):
        """Calculate a confidence score for spam classification"""
        confidence = 0
        if patterns.get('immediate_response'):
            confidence += 0.5
        if patterns.get('repetitive_content'):
            confidence += 0.6
        if patterns.get('unknown_number'):
            confidence += 0.4
        
        return min(confidence, 1.0)

# Example usage
if __name__ == "__main__":
    user_contacts = {'+1234567890', '+1987654321'}
    analyzer = CallPatternAnalyzer(user_contacts, time_window=24)
    
    spam_call = {
        'caller_number': '+1122334455',
        'transcript': 'Congratulations! You have won a free vacation! Press 1 to claim.',
        'response_time': 0.4,
        'timestamp': datetime.now()
    }
    
    spam_call_repeat = {
        'caller_number': '+1122334455',
        'transcript': 'Congratulations! Free vacation! Press 1 now!',
        'response_time': 0.5,
        'timestamp': datetime.now() + timedelta(minutes=15)
    }
    
    legit_call = {
        'caller_number': '+1234567890',
        'transcript': 'Hey Mike, just checking in about our meeting tomorrow.',
        'response_time': 2.2,
        'timestamp': datetime.now()
    }
    
    print("First spam call analysis:")
    print(analyzer.analyze_call_patterns(spam_call))
    
    print("\nRepeat spam call analysis:")
    print(analyzer.analyze_call_patterns(spam_call_repeat))
    
    print("\nLegitimate call analysis:")
    print(analyzer.analyze_call_patterns(legit_call))
