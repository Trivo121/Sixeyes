def load_keywords(keywords_file):
    try:
        with open(keywords_file, 'r', encoding='utf-8') as f:
            keywords = [line.strip().lower() for line in f]
        return keywords
    except Exception as e:
        raise RuntimeError(f"Error loading keywords from {keywords_file}: {e}")
