from spam_utils import preprocess_message

def test_preprocess_strips_and_lowercases_text():
    assert preprocess_message("  HeLLo WoRLD  ") == "hello world"

def test_preprocess_empty_string():
    assert preprocess_message("") == ""

def test_preprocess_none_returns_empty_string():
    assert preprocess_message(None) == ""