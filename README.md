---
title: Scholar Lens
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: mit
---

# Scholar Lens

Scholar Lens is a Gradio research assistant for searching papers across Semantic Scholar, arXiv, and PubMed from one focused interface. It normalizes results into a clean table, lets users select a paper, and sends the title plus abstract directly to an AI summarizer hosted on Modal.

## Features

- Search Semantic Scholar, arXiv, and PubMed with one query.
- View normalized paper metadata: title, year, source, authors, citations, and abstract availability.
- Select a search result and load it into the Summarize tab.
- Summarize selected papers or pasted abstracts with a Modal-hosted language model.
- Clear search results and reset the summarization workspace.
- Professional dark Gradio interface with responsive result tables and source badges.

## Codex Track

This project was built with OpenAI Codex as the coding agent.

Public GitHub repository: add your final public repo link here before submission.

## Project Structure

```text
app.py              Main Gradio application
modal_inference.py  Modal endpoint for AI summarization
requirements.txt    Python dependencies
SPEC.md             Initial project specification
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
python app.py
```

The app launches a local Gradio server and prints the URL in the terminal.

## Modal Summarizer

`modal_inference.py` defines a Modal FastAPI endpoint backed by `Qwen/Qwen2.5-7B-Instruct`. The Gradio app posts paper text to the deployed Modal endpoint configured in `MODAL_SUMMARIZE_URL`.

If you deploy your own Modal endpoint, update `MODAL_SUMMARIZE_URL` in `app.py`.
