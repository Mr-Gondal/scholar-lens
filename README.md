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

Scholar Lens is a Gradio research assistant for searching papers across OpenAlex, Crossref, arXiv, and PubMed from one focused interface. It normalizes and de-duplicates results into a clean table, lets users select a paper, and sends the title plus abstract to a strong open language model (Mistral Small 3.1 24B) hosted on Modal.

## Features

- **Ask (grounded Q&A):** ask a research question; the app searches all four sources and a 24B open model writes a synthesized, **cited** answer grounded only in the retrieved abstracts (no invented sources).
- Search OpenAlex, Crossref, arXiv, and PubMed with one query.
- De-duplicates results across sources by DOI, then normalized title.
- View normalized paper metadata: title, year, source, authors, citations, and abstract availability.
- Select a search result and load it into the Summarize tab.
- Summarize selected papers or pasted abstracts with a Modal-hosted language model.
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

## Modal Inference

`modal_inference.py` defines a Modal class backed by `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (a 24B, <=32B open model). The model is loaded once per container via `@modal.enter()` and kept warm briefly for demos, then exposed through two FastAPI endpoints:

- `summarize_paper` — condenses a single abstract or pasted text (`MODAL_SUMMARIZE_URL`).
- `synthesize` — answers a question grounded in numbered abstracts, with `[n]` citations (`MODAL_SYNTHESIZE_URL`).

Create a shared API token and store the same value in both places:

- In your Gradio/Hugging Face Space environment: `SCHOLAR_LENS_MODAL_TOKEN`.
- In Modal, create a secret named `scholar-lens-api` with `SCHOLAR_LENS_MODAL_TOKEN`.

Deploy with:

```bash
modal deploy modal_inference.py
```

After deployment, set these Gradio/Hugging Face Space environment variables to the URLs Modal prints:

```text
MODAL_SUMMARIZE_URL=https://your-summarize-endpoint.modal.run
MODAL_SYNTHESIZE_URL=https://your-synthesize-endpoint.modal.run
SCHOLAR_LENS_MODAL_TOKEN=your-shared-secret-token
```

The Modal endpoints reject requests that do not include the matching `Authorization: Bearer <token>` header.

Input protection:

- Summarization text is capped before it reaches Modal. Longer valid inputs are summarized in bounded chunks, then combined into a final summary.
- Grounded synthesis uses a fixed context budget so retrieved abstracts cannot overflow the model context window.
- The app also applies textbox limits in the UI and shows a friendly message when input is too large.
