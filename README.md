## Sixeyes
      Sixeyes is an AI model which is mainly focused to detect the spam calls and recognize fraud calls in realtime . ik there are many apps which could recognize  spamcalls but we are trying to make this one more accurate and could be viable in realtime. we want to build a system at which it should spam a calls through the previous history database and it should recognize the calls to be based on their behaviour and anomaly patterns. the goal and logic was defined clearly for us. but the whole project is at development phase we are still in the development phase at this level we could only load and preprocess the data, extract text from it, and reads the transcript and tells whether it's a spam/fraud call or not. we need to take it up a bit high to implemt this in real time. that is still in development

## Features

### 1. Spam/Fraud Call Detection (In Development)
- Real-time audio analysis
- NLP-based classification using DistilBERT
- Voice pattern recognition

### 2. Deepfake Video Recognition (Completed) ✔️
- VKYC (Video Know Your Customer) integration
- Real-time frame analysis
- Pre-trained classification models

### 3. Feedback & Monitoring System (Pending)
- User-reported spam numbers database
- Community-driven fraud prevention
- Continuous learning from user feedback

## Current Progress

### Completed Components
- **Deepfake Detection System**
  - Implemented frame-level analysis pipeline
  - Integrated YOLOv3 for object consistency checks
  - Developed OpenCV-based video processing workflow

- **Spam Detection Foundation**
  - Completed audio preprocessing pipeline
  - Implemented Librosa-based feature extraction
  - Set up DistilBERT text classification skeleton
  - Created initial dataset structure

### Technical Achievements
- Successful integration of Whisper Small for ASR
- Optimized audio preprocessing with Librosa
- Established baseline metrics for deepfake detection

## Pending Features

### Spam/Fraud Detection
- [ ] DistilBERT model training/evaluation
- [ ] Real-time audio stream integration
- [ ] Voice signature analysis component
- [ ] Multi-factor authentication logic

### Feedback System
- [ ] User reporting API endpoints
- [ ] Community trust scoring mechanism
- [ ] Feedback-based model retraining pipeline

### General
- [ ] End-to-end encryption implementation
- [ ] Performance optimization for edge devices
- [ ] Cross-platform compatibility layer

## Tech Stack

### Core AI Components
- **Spam Detection**
  - Python, PyTorch
  - Transformers (DistilBERT)
  - Librosa, OpenAI Whisper
  - NLTK, Pandas

- **Deepfake Detection**
  - OpenCV, YOLOv3
  - Pretrained CNN models
  - FFmpeg video processing

### Infrastructure
- Hugging Face Datasets
- Jupyter Notebooks
- GitHub Actions (CI/CD)

