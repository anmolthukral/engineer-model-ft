#!/usr/bin/env python3
"""Minimal local chat UI for the RAG-augmented tone model. Reuses the same
retrieve()/build_prompt() logic rag/query.py uses for the CLI, so the UI
and the CLI share one source of truth for the RAG flow instead of
duplicating it."""
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag.query import retrieve, build_prompt, OLLAMA_GENERATE_URL, TONE_MODEL

st.set_page_config(page_title="Engineering Tutor", page_icon="🎓")
st.title("Engineering Tutor")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about JS, React, Next.js, or TypeScript...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Looking up docs and thinking..."):
            chunks = retrieve(question)
            prompt = build_prompt(question, chunks)
            # raw=True: prompt already has the full "### User:/### Assistant:"
            # markers baked in — see the note in rag/query.py's ask().
            resp = requests.post(
                OLLAMA_GENERATE_URL,
                json={"model": TONE_MODEL, "prompt": prompt, "stream": False, "raw": True},
            )
            resp.raise_for_status()
            answer = resp.json()["response"]
        st.markdown(answer)
        with st.expander("Sources"):
            for c in chunks:
                st.caption(f"{c['source']}: {c['file_path']} ({c['heading_trail']})")

    st.session_state.messages.append({"role": "assistant", "content": answer})
