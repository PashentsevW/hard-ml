from typing import List

import nltk
from nltk.stem import WordNetLemmatizer


def document_tokenize(document: str) -> List[str]:
    if not document:
        return []

    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(token.lower())
            for token in nltk.word_tokenize(document)
            if token.isalnum()]
