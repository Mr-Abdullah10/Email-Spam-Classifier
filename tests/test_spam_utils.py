from spam_utils import preprocess_message

def test_preprocess_strips_leading_trailing_whitespace():
    assert preprocess_message("  hello world  ") == "hello world"

def test_preprocess_empty_string():
    assert preprocess_message("") == ""

def test_preprocess_none_returns_empty_string():
    assert preprocess_message(None) == ""