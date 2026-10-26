import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def remove_punctuation(message):
    message = message.lower()
    result = []

    for char in message:
        if char not in string.punctuation:
            result.append(char)

    return "".join(result)


def remove_stopword(tokens):
    return [word for word in tokens if word not in STOP_WORDS]


def lemmatized_message(tokens):
    return [LEMMATIZER.lemmatize(word) for word in tokens]


def preprocess_message(message):
    cleaned = remove_punctuation(message)
    tokens = cleaned.split()
    tokens = remove_stopword(tokens)
    tokens = lemmatized_message(tokens)

    return " ".join(tokens)