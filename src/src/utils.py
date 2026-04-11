import re

def clean_text(text):
    # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # convert to lowercase
    text = text.lower()

    return text