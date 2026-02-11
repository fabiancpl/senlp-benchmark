"""Language detector tool.

This module implements a FastText inference process for detecting the language of a given text. It
is based on a pre-built model available on the FastText official website.
"""

import re

import fasttext


# The model binary can be downloaded here:
# https://fasttext.cc/docs/en/language-identification.html
MODEL_PATH = "../../assets/models/lid.176.bin"
model = fasttext.load_model(MODEL_PATH)


def detect_lang(text: str, k: int = 3) -> list[tuple[str, float]]:
    """Returns the top k languages and scores for the input string"""
    if (text is None) or (text == ""):
        return []

    try:
        text = re.sub(r'[\r\n]+', " ", text)
        preds = model.predict(text, k=k)

        preds_mod = []
        for lang, score in zip(preds[0], preds[1]):
            preds_mod.append((lang.replace("__label__", ""), score))

        return preds_mod
    except TypeError:
        return []
