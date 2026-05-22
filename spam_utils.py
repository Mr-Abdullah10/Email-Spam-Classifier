def preprocess_message(text: str) -> str:
    """Prepare message text before vectorization."""
    if text is None:
        return ""
    return text.strip()