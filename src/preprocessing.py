"""
Text Preprocessing Module for YouTube Spam Comments.
Preserves crucial spam indicators (URLs, channel callouts, special symbols)
while standardizing HTML entities, whitespace, and formatting.
"""

import re
import html
from typing import Optional

# Regex pattern for URLs (http, https, www, or typical domains)
URL_PATTERN = re.compile(
    r'(https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|io|co|me|tv|gl|ly|be|app)\b\S*)',
    re.IGNORECASE
)

# Regex pattern for HTML tags
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

# Regex pattern for repetitive whitespace
WHITESPACE_PATTERN = re.compile(r'\s+')

def clean_comment_text(
    text: Optional[str],
    normalize_urls: bool = True,
    url_token: str = "http_url",
    remove_html_tags: bool = True,
    unescape_html: bool = True
) -> str:
    """
    Cleans raw comment text while preserving spam signals.
    
    Args:
        text: Raw comment string.
        normalize_urls: Whether to standardize URLs to a normalized token.
        url_token: Token to replace URLs with.
        remove_html_tags: Whether to strip raw HTML tags like <br>, <p>.
        unescape_html: Whether to unescape HTML entities like &amp;, &#39;.
        
    Returns:
        Cleaned and normalized comment string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Unescape HTML entities (&amp; -> &, &quot; -> ", etc.)
    if unescape_html:
        text = html.unescape(text)

    # Remove residual HTML markup if any
    if remove_html_tags:
        text = HTML_TAG_PATTERN.sub(" ", text)

    # Normalize URLs consistently
    if normalize_urls:
        text = URL_PATTERN.sub(f" {url_token} ", text)

    # Collapse multiple whitespaces and trim edges
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text
