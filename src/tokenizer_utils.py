'''
import tensorflow as tf


def fit_vectorizer(text,MAX_LEN):
  vectorizer = tf.keras.layers.TextVectorization(
    output_sequence_length=MAX_LEN,
    )
  vectorizer.adapt(text)
  return vectorizer
'''
import tensorflow as tf
import re

# Custom standardization that preserves <start> and <end>
def custom_standardization(input_string):
    # Lowercase
    lowercase = tf.strings.lower(input_string)
    # Remove URLs, mentions, and keep only certain characters (preserving <, >)
    # Optionally, you can add more regex as you need
    cleaned = tf.strings.regex_replace(lowercase, r"http\S+", "")
    cleaned = tf.strings.regex_replace(cleaned, r"@\w+", "")
    cleaned = tf.strings.regex_replace(cleaned, r"[^a-z0-9\s<>]", "")
    cleaned = tf.strings.regex_replace(cleaned, r"\s+", " ")
    return cleaned

def fit_vectorizer(text, MAX_LEN):
    vectorizer = tf.keras.layers.TextVectorization(
        output_sequence_length=MAX_LEN,
        standardize=custom_standardization,  # <-- Key part!
    )
    vectorizer.adapt(text)
    return vectorizer
