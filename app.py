from __future__ import annotations

from typing import Dict, List

import streamlit as st

from config import get_settings
from ollama_client import OllamaChatClient
from parser import fetch_and_extract
from rag import build_context_messages


def get_chat_history() -> List[Dict[str, str]]:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages


def main() -> None:
    st.set_page_config(page_title="Ollama Chat", page_icon="🤖", layout="centered")
    settings = get_settings()

    st.sidebar.title("Settings")
    st.sidebar.write(f"Model: `{settings.model}`")
    st.sidebar.write(f"Host: `{settings.ollama_host}`")
    with st.sidebar.expander("Performance"):
        num_predict = st.number_input("Max tokens (num_predict)", min_value=64, max_value=8192, value=settings.num_predict, step=64)
        num_ctx = st.number_input("Context size (num_ctx)", min_value=512, max_value=32768, value=settings.num_ctx, step=256)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(settings.temperature), step=0.05)
        top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=float(settings.top_p), step=0.05)
        num_thread = st.number_input("Threads", min_value=1, max_value=64, value=settings.num_thread, step=1)
        stream_batch = st.number_input("UI token batch size", min_value=1, max_value=200, value=settings.stream_token_batch_size, step=1)
        keep_alive = st.text_input("Keep alive", value=settings.keep_alive, help="How long to keep the model in memory (e.g., 10m, 1h)")
    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.sidebar.caption("Change via env vars: OLLAMA_MODEL, OLLAMA_HOST, SYSTEM_PROMPT")

    st.title("Chat with the AI model")
    st.caption("Streaming responses from your local model. Type below to start.")
    mode = st.radio("Mode", ["Chat", "RAG"], horizontal=True)

    # Ensure RAG context persists across reruns
    if "rag_context_messages" not in st.session_state:
        st.session_state.rag_context_messages = []
    if "rag_context_meta" not in st.session_state:
        st.session_state.rag_context_meta = {}
    if "rag_messages_in_history" not in st.session_state:
        st.session_state.rag_messages_in_history = False
    if "rag_raw_text" not in st.session_state:
        st.session_state.rag_raw_text = ""

    client = OllamaChatClient(host=settings.ollama_host, model=settings.model)

    messages = get_chat_history()
    if not messages:
        # Seed with system prompt if starting fresh
        messages.append({"role": "system", "content": settings.system_prompt})

    # Render existing conversation (skip system prompt)
    for msg in messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"]) 

    rag_url = None
    if mode == "RAG":
        with st.expander("RAG options"):
            rag_url = st.text_input("URL to fetch")
            rag_max_lines = st.number_input("Max lines from page", min_value=100, max_value=5000, value=1200, step=100)
            inline_with_question = st.checkbox(
                "Inline article context with your question (recommended)",
                value=True,
                help="Pastes the article context into the same prompt as your question so the model definitely sees it.",
                key="inline_ctx_checkbox",
            )
            if st.button("Fetch page") and rag_url:
                try:
                    with st.spinner("Fetching and parsing page…"):
                        page_text, meta = fetch_and_extract(rag_url, max_lines=int(rag_max_lines))
                    ctx = build_context_messages(page_text, meta)
                    st.session_state.rag_context_messages = ctx
                    st.session_state.rag_context_meta = meta
                    st.session_state.rag_raw_text = page_text
                    # Inject context system messages into conversation history immediately
                    # Keep initial system prompt(s) at the start
                    system_msgs = [m for m in st.session_state.messages if m.get("role") == "system"]
                    non_system = [m for m in st.session_state.messages if m.get("role") != "system"]
                    st.session_state.messages = system_msgs + ctx + non_system
                    st.session_state.rag_messages_in_history = True
                    st.success(f"Fetched {meta.get('lines')} lines: {meta.get('title','')}")
                except Exception as e:
                    st.error(f"Failed to fetch: {e}")
        if st.session_state.rag_context_messages:
            meta = st.session_state.get("rag_context_meta", {})
            title = meta.get("title", "")
            source = meta.get("source_url", "")
            st.success(f"Using article context{f' — {title}' if title else ''}")
            if source:
                st.caption(f"Source: {source}")

    user_input = st.chat_input("Send a message…")
    if user_input:
        # Show the user's new message immediately
        messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Use messages directly if context was injected; otherwise, prepend on the fly
        full_messages = messages
        if mode == "RAG" and st.session_state.rag_context_messages and not st.session_state.get("rag_messages_in_history", False):
            system_msgs = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]
            full_messages = system_msgs + st.session_state.rag_context_messages + non_system

        # Stream assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            accumulated = ""
            buffer = []
            options = {
                "num_predict": int(num_predict),
                "num_ctx": int(num_ctx),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "num_thread": int(num_thread),
            }
            # Optionally inline article context into the same user prompt being sent now
            send_messages = full_messages
            if mode == "RAG" and st.session_state.get("inline_ctx_checkbox", True) and st.session_state.get("rag_raw_text"):
                # Replace the last user message with a combined prompt that includes context
                # Find the last user message index
                last_user_idx = None
                for i in range(len(full_messages) - 1, -1, -1):
                    if full_messages[i].get("role") == "user":
                        last_user_idx = i
                        break
                if last_user_idx is not None:
                    question = full_messages[last_user_idx].get("content", "")
                    meta = st.session_state.get("rag_context_meta", {})
                    title = meta.get("title", "")
                    source = meta.get("source_url", "")
                    context_header = "Article context provided below. Use it to answer the question."
                    if title:
                        context_header += f"\nTitle: {title}"
                    if source:
                        context_header += f"\nSource: {source}"
                    combined = (
                        f"{context_header}\n[BEGIN ARTICLE CONTEXT]\n{st.session_state.rag_raw_text}\n[END ARTICLE CONTEXT]\n\n"
                        f"Question: {question}"
                    )
                    send_messages = list(full_messages)
                    send_messages[last_user_idx] = {"role": "user", "content": combined}

            for token in client.generate(
                send_messages,
                stream=True,
                options=options,
                keep_alive=keep_alive,
            ):
                # Stream every token (or per UI batch size if user adjusts)
                buffer.append(token)
                if len(buffer) >= int(stream_batch):
                    accumulated += "".join(buffer)
                    buffer.clear()
                    placeholder.markdown(accumulated)
            if buffer:
                accumulated += "".join(buffer)
                placeholder.markdown(accumulated)
            messages.append({"role": "assistant", "content": accumulated})


if __name__ == "__main__":
    main()


