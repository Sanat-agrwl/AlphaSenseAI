# Lazy imports — transformers/torch/whisper are heavy and only loaded on first use.
__all__ = [
    "TextSentimentPipeline", "aggregate_sentiment_by_stock", "load_scored_news",
    "AudioPipeline", "ConfidenceDeltaDetector",
]