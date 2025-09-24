from __future__ import annotations

from typing import Dict, Generator, Iterable, List, Optional

import ollama


Message = Dict[str, str]


class OllamaChatClient:
    def __init__(self, host: str, model: str) -> None:
        self._client = ollama.Client(host=host)
        self.model = model

    def generate(
        self,
        messages: List[Message],
        stream: bool = True,
        options: Optional[Dict] = None,
        keep_alive: Optional[str] = None,
    ) -> Iterable[str] | str:
        params = {"model": self.model, "messages": messages, "stream": stream}
        if options:
            params["options"] = options
        if keep_alive is not None:
            params["keep_alive"] = keep_alive

        if stream:
            return self._stream(params)
        result = self._client.chat(**params)
        return result.get("message", {}).get("content", "")

    def _stream(self, params: Dict) -> Generator[str, None, None]:
        stream = self._client.chat(**params)
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content


