from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
def analyze_sentiment(text):
    """Analyze sentiment of the input text using a Hugging Face model."""
    result = sentiment_pipeline(text)
    return result[0]
if __name__ == "__main__":
    while True:
        text = input("Enter a sentence for sentiment analysis (or 'exit' to quit): ")
        if text.lower() == "exit":
            break
        sentiment_result = analyze_sentiment(text)
        print(f"Sentiment: {sentiment_result['label']} (Confidence: {sentiment_result['score']:.2f})\n")
