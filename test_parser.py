from __future__ import annotations

import sys
from textwrap import shorten

from parser import fetch_and_extract


def main() -> None:
    # Default URL for quick test; can be overridden via CLI arg
    default_url = "https://r.jina.ai/https://www.themarginalian.org/2014/01/03/baloney-detection-kit-carl-sagan/"
    url = sys.argv[1] if len(sys.argv) > 1 else default_url
    max_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 800

    print(f"Fetching: {url}")
    content, meta = fetch_and_extract(url, max_lines=max_lines)

    title = meta.get("title", "")
    source = meta.get("source_url", "")
    lines = int(meta.get("lines", "0"))

    print("\n=== Metadata ===")
    print(f"Title: {title}")
    print(f"Source: {source}")
    print(f"Lines: {lines}")

    print("\n=== Preview (first 40 lines or fewer) ===")
    preview_lines = content.splitlines()[:40]
    for i, line in enumerate(preview_lines, start=1):
        print(f"{i:02d}: {line}")

    print("\n=== One-liner summary (truncated) ===")
    print(shorten(" ".join(preview_lines), width=220, placeholder="…"))


if __name__ == "__main__":
    main()


