import pandas as pd
import numpy as np
import torch
import re
from typing import Dict, List
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from utils.file_handler import save_data
from utils.logger import setup_logger
from utils.config import CONFIG
from concurrent.futures import ProcessPoolExecutor

logger = setup_logger('feature_extraction')

class FeatureEngineer:
    """Enhanced feature engineering with DistilBERT integration"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = DistilBertTokenizer.from_pretrained(config['MODEL_PATH'])
        self.model = DistilBertForSequenceClassification.from_pretrained(config['MODEL_PATH'])
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.batch_size = config.get('BERT_BATCH_SIZE', 8)

    def _extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract call metadata features"""
        df['call_frequency'] = df.groupby('caller_id')['timestamp'].transform('count')
        df['call_duration'] = df['call_duration'].dt.total_seconds()
        df['hour_of_day'] = df['timestamp'].dt.hour
        return df

    def _process_text_batch(self, texts: List[str]) -> List[Dict]:
        """Process batch of texts with DistilBERT"""
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        batch_features = []
        
        for i in range(len(texts)):
            # Extract attention for keywords
            attention = outputs.attentions[-1][i].mean(dim=0).mean(dim=0).cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])
            
            keyword_scores = {
                token: score for token, score in zip(tokens, attention)
                if token not in ['[CLS]', '[SEP]', '[PAD]']
            }
            top_keywords = sorted(
                keyword_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            batch_features.append({
                'spam_score': probs[i][1],
                'top_keywords': [kw[0] for kw in top_keywords],
                'attention_variance': np.var(attention)
            })
        
        return batch_features

    def process(self):
        """Main processing pipeline"""
        try:
            features = []
            for batch in pd.read_csv(
                CONFIG['PROCESSED_DATA_PATH'],
                chunksize=CONFIG['BATCH_SIZE'],
                parse_dates=['timestamp']
            ):
                # Metadata features
                meta_df = self._extract_metadata_features(batch)
                
                # Text processing
                texts = meta_df['transcript'].fillna('').tolist()
                text_features = []
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:i+self.batch_size]
                    text_features.extend(self._process_text_batch(batch_texts))
                
                # Combine features
                final_df = pd.concat([
                    meta_df,
                    pd.DataFrame(text_features)
                ], axis=1)
                
                features.append(final_df)
            
            # Save combined features
            full_features = pd.concat(features)
            save_data(full_features, CONFIG['FEATURE_DATA_PATH'])
            logger.info(f"Features saved to {CONFIG['FEATURE_DATA_PATH']}")

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

if __name__ == '__main__':
    engineer = FeatureEngineer(CONFIG)
    engineer.process()