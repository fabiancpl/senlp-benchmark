"""Text pre-processing steps.

This module implements all the required steps for text pre-processing following the Scikit-Learn
Pipeline interface.
"""

import re

from bs4 import BeautifulSoup
import mistune
from sklearn.base import BaseEstimator, TransformerMixin
import swifter  # noqa: F401  # pylint: disable=unused-import


class BasicPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lower=True, remove_newlines=False):
        self.lower = lower
        self.remove_newlines = remove_newlines

        self.newline_pattern = re.compile(r'[\r\n]+')
        self.whitespace_pattern = re.compile(r'\s+')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Lowercase
        if self.lower:
            X = X.str.lower()

        # Trim whitespaces
        X = X.str.strip()

        # Remove new lines
        if self.remove_newlines:
            X = X.str.replace(self.newline_pattern, " ", regex=True)

        # Normalize whitespaces
        X = X.str.replace(self.whitespace_pattern, " ", regex=True)

        return X


class HTMLExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, in_markdown: bool = False, strip_special_tags: bool = False):
        self.in_markdown = in_markdown
        self.strip_special_tags = strip_special_tags

    def _extract(self, text: str | None) -> str | None:
        if text is None:
            return None

        if self.in_markdown:
            # Markdown to HTML
            text = mistune.html(text)

        try:
            # Extract natural text from HTML
            soup = BeautifulSoup(text, "html.parser")
            soup = self._mask(soup)
            return soup.get_text().strip()
        except Exception:
            return None

    def _mask(self, soup: BeautifulSoup) -> BeautifulSoup:
        for code_tag in soup.find_all("code"):
            code_tag.replace_with("[CODE]" if not self.strip_special_tags else "")

        for url_tag in soup.find_all("a"):
            url_tag.replace_with("[URL]" if not self.strip_special_tags else "")

        return soup

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.swifter.progress_bar(desc="Cleaning HTML tags").apply(self._extract)


class Jira2MarkdownConverter(BaseEstimator, TransformerMixin):
    def _convert(self, text: str) -> str | None:
        if text is None:
            return None

        # TODO: Debug on pre-training datasets
        # Bold
        text = re.sub(r'\*(.*?)\*', r'**\1**', text)
        # Italics
        text = re.sub(r'\_(.*?)\_', r'*\1*', text)
        # Monospaced text
        text = re.sub(r'\{\{(.*?)\}\}', r'`\1`', text)
        # Headings
        text = re.sub(r'h([1-6])\.(.*)', lambda m: '#' * int(m.group(1)) + m.group(2), text)
        # Bulleted lists
        text = re.sub(r'^\s*\*(.*)', r'-\1', text, flags=re.MULTILINE)
        # Numbered lists
        text = re.sub(r'^\s*#(.*)', r'1.\1', text, flags=re.MULTILINE)
        # Blockquote
        text = re.sub(r'\{quote\}(.*?)\{quote\}', r'> \1', text, flags=re.DOTALL)
        # Links
        text = re.sub(r'\[(.*?)\|(.*?)\]', r'[\1](\2)', text)
        # Code blocks
        text = re.sub(r'\{code\}(.*?)\{code\}', r'```\1```', text, flags=re.DOTALL)
        text = re.sub(r'\{noformat\}(.*?)\{noformat\}', r'```\1```', text, flags=re.DOTALL)

        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.swifter.progress_bar(desc="Cleaning JIRA tags").apply(self._convert)


class RegexMasker(BaseEstimator, TransformerMixin):
    def __init__(self, pattern: str, strip: bool = False):
        if pattern == "CODE":
            self.pattern = re.compile(r'```[\s\S]*?```')
        elif pattern == "HASH":
            self.pattern = re.compile(r'\b[a-fA-F0-9]{40}\b')
        elif pattern == "URL":
            self.pattern = re.compile(
                r'https?:\s*//(?:[-\w.]|(?:%[\da-fA-F]{2}))+(/[^\s"()]*)?'
            )
        elif pattern == "USER":
            self.pattern = re.compile(r'(?<![@\w])@(\w{1,38})')
        elif pattern == "NON-UTF8":
            self.pattern = re.compile(r'[^\x00-\x7F]+')
        else:
            raise ValueError("A valid Regex pattern must be provided")

        self.mask = f"[{pattern}]" if not strip else ""

    def _mask(self, text: str | None) -> str | None:
        if text is None:
            return None

        return self.pattern.sub(self.mask, text)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.swifter.progress_bar(desc=f"Masking {self.mask}").apply(self._mask)
