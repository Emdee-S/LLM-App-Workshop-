from __future__ import annotations

from typing import Dict, List


def build_context_messages(page_text: str, metadata: Dict[str, str]) -> List[Dict[str, str]]:
    """Create explicit system messages instructing the model to use the article context.

    Returns two system messages: an instruction and the context payload with clear markers.
    """
    source = metadata.get("source_url", "")
    title = metadata.get("title", "")
    lines = metadata.get("lines", "")

    instruction = (
        "You are provided with article context. The user will ask a question about it next. "
        "Use the article context to answer directly and concisely. If the answer is not in the "
        "context, say you don't know. Prefer quoting or referencing exact phrases when helpful."
    )

    meta_lines = []
    if title:
        meta_lines.append(f"Title: {title}")
    if source:
        meta_lines.append(f"Source: {source}")
    if lines:
        meta_lines.append(f"Lines: {lines}")
    meta_header = "\n".join(meta_lines)

    context_payload = (
        (meta_header + "\n") if meta_header else ""
        ) + "[BEGIN ARTICLE CONTEXT]\n" + page_text + "\n[END ARTICLE CONTEXT]"

    return [
        {"role": "system", "content": instruction},
        {"role": "system", "content": context_payload},
    ]


