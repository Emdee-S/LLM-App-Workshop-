from __future__ import annotations

import re
from typing import Dict, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _clean_text(text: str) -> str:
    # Normalize whitespace and collapse multiple blank lines
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\u00A0", " ", text)  # non-breaking space
    # Collapse >1 spaces
    text = re.sub(r"[ \f\v]+", " ", text)
    # Keep single newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_MD_IMG_PATTERN = re.compile(r"!\[[^\]]*\]\([^\)]+\)")
_MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")
_URL_PATTERN = re.compile(r"https?://\S+")


def _strip_markdown_links(text: str) -> str:
    """Remove markdown images, convert markdown links to plain text, and drop bare URLs.

    Examples:
    - ![Alt](http://img) -> ''
    - [Label](http://link) -> 'Label'
    - https://example.com -> ''
    """
    text = _MD_IMG_PATTERN.sub("", text)
    text = _MD_LINK_PATTERN.sub(r"\1", text)
    text = _URL_PATTERN.sub("", text)
    # Collapse extra spaces left by removals
    text = re.sub(r"\s{2,}", " ", text)
    return text


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _request_html(session: requests.Session, url: str, timeout: int) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    resp = session.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        # Still attempt to parse as text, but sites may set odd types
        pass
    return resp.text


def fetch_and_extract(url: str, max_lines: int = 800, timeout: int = 20) -> Tuple[str, Dict[str, str]]:
    """Fetch a URL and extract readable text.

    Returns (content, metadata). Content is limited to max_lines.
    """
    session = _build_session()
    try:
        html = _request_html(session, url, timeout=timeout)
    except Exception:
        # Fallback via Jina Reader proxy to avoid origin anti-bot resets
        prefix = "https://r.jina.ai/"
        proxied_url = url
        if not url.startswith(prefix):
            proxied_url = prefix + url
        html = _request_html(session, proxied_url, timeout=timeout)

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "footer", "nav"]):
        tag.extract()

    # Prefer article/main when available
    container = soup.find("article") or soup.find("main") or soup.body or soup
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    text = container.get_text("\n")
    text = _clean_text(text)
    # If source was proxied via reader (markdown-like), strip links/images and bare URLs
    text = _strip_markdown_links(text)

    # Split to lines, trim and cap
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if max_lines > 0:
        lines = lines[:max_lines]
    content = "\n".join(lines)

    meta: Dict[str, str] = {
        "title": title,
        "source_url": url,
        "lines": str(len(lines)),
    }
    return content, meta


