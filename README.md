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
tags:
  - build-small-hackathon
  - backyard-ai
  - nvidia-nemotron
  - openai-codex
  - modal
---

# Scholar Lens

Scholar Lens turns a research question into a cited, cross-database answer using NVIDIA Nemotron on Modal. It is built for an atmospheric-science professor who needs to triage papers across OpenAlex, Crossref, arXiv, and PubMed without copying abstracts between tools.

The app normalizes and de-duplicates results into a clean table, lets users inspect or summarize individual papers, and uses `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` on Modal for grounded synthesis over retrieved abstracts.

## Features

- **Ask (grounded Q&A):** ask a research question; the app searches all four sources and NVIDIA Llama-Nemotron-Nano 8B writes a synthesized, **cited** answer grounded only in the retrieved abstracts (no invented sources).
- **Compare with Nemotron:** select two papers and get a grounded NVIDIA Nemotron comparison of methods, claims, limitations, and best use cases.
- Atmospheric-science starter questions for aerosol-cloud interactions, satellite precipitation, atmospheric rivers, and urban heat islands.
- Search OpenAlex, Crossref, arXiv, and PubMed with one query.
- De-duplicates results across sources by DOI, then normalized title.
- View normalized paper metadata: title, year, source, authors, citations, and abstract availability.
- Select a search result and load it into the Summarize tab.
- Summarize selected papers or pasted abstracts with a Modal-hosted language model.
- View search insight cards, evidence snippets, and export results or summaries.
- Professional dark Gradio interface with responsive result tables and source badges.

## Codex Track

This project was built with OpenAI Codex as the coding agent. The commit history is intentionally small and reviewable so the build process can be evaluated alongside the app.

Hackathon pitch: Scholar Lens turns a research question into a cited, cross-database answer using a model small enough to run on a single consumer GPU, built for a real atmospheric-science researcher who was drowning in literature review.

Public GitHub repository: https://github.com/Mr-Gondal/scholar-lens

Demo video: _add your demo video link here before submission._

Social post (REQ-04): _add your X/LinkedIn post link here before submission._

## Backyard AI Rubric Fit

| Judging criterion | Scholar Lens proof |
|---|---|
| Problem specificity and reality | The app is framed around one atmospheric-science professor's literature-review workflow: scattered databases, repeated abstract triage, and slow cross-paper synthesis. |
| Actual user adoption | Validated with a real atmospheric-science researcher running their own research questions; the demo includes their usage and a short quote. |
| Honest small-model fit | NVIDIA Llama-Nemotron-Nano 8B is load-bearing because the task is grounded synthesis over retrieved abstracts. If the model is removed, Scholar Lens becomes a list of links. |
| NVIDIA prize fit | The main demo path uses **Powered by NVIDIA Nemotron on Modal** for grounded paper comparison and synthesis. |
| App polish | Ask is the default path, sources are cited, references are linked, search is paginated, and empty/error states are designed for a judge's first click. |

## Project Structure

```text
app.py              Main Gradio application
modal_inference.py  Modal (cloud) inference endpoints
local_inference.py  Cloud-free llama.cpp / GGUF path (Off-Grid, benchmarks)
requirements.txt    Python dependencies
FIELD_NOTES.md      Build report (problem, decisions, benchmarks)
SPEC.md             Initial project specification
tests/              Unit tests for core logic
```

## Benchmarks & Local Inference

Scholar Lens is small enough to run with **no cloud APIs** on a single consumer
GPU via `local_inference.py` (llama.cpp / GGUF) — this is the basis for the
Off-Grid and Llama Champion merit badges.

```bash
pip install llama-cpp-python
export SCHOLAR_LENS_GGUF=/path/to/qwen2.5-3b-instruct-q4_k_m.gguf
python local_inference.py --benchmark
```

| Setup | Model / quant | VRAM | Throughput (tok/s) | Ask latency (s) |
|---|---|---|---|---|
| Modal (cloud) | NVIDIA Llama-Nemotron-Nano 8B / bf16 | — | — | — |
| Local GPU | Qwen2.5-3B / Q4_K_M | — | — | — |
| Local GPU | Qwen2.5-1.5B / Q4_K_M | — | — | — |

*Fill the dashes from a real run; the smaller the model that holds answer
quality, the stronger the small-model and consumer-GPU story.*

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

Optional local configuration:

```bash
set SCHOLAR_LENS_CONTACT_EMAIL=you@example.com
set MODAL_SUMMARIZE_URL=https://your-summarize-endpoint.modal.run
set MODAL_SYNTHESIZE_URL=https://your-synthesize-endpoint.modal.run
set SCHOLAR_LENS_MODAL_TOKEN=your-shared-secret-token
```

Run tests:

```bash
python -B -m unittest discover -s tests -v
```

## Modal Inference

`modal_inference.py` defines a Modal class backed by `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` by default. The model is loaded once per container via `@modal.enter()` and kept warm briefly for demos, then exposed through two FastAPI endpoints:

The default model is intentionally the NVIDIA Llama-Nemotron-Nano 8B release: it is NVIDIA Nemotron (for the NVIDIA prize), uses the standard Llama architecture so it loads on stock vLLM on a single GPU, and supports the `detailed thinking off` reasoning toggle for grounded, low-latency answers. Larger Nemotron releases can be tested later by overriding `SCHOLAR_LENS_MODEL` and `SCHOLAR_LENS_GPU`.

- `summarize_paper` — condenses a single abstract or pasted text (`MODAL_SUMMARIZE_URL`).
- `synthesize` — answers a question grounded in numbered abstracts, with `[n]` citations (`MODAL_SYNTHESIZE_URL`).

Create a shared API token and store the same value in both places:

- In your Gradio/Hugging Face Space environment: `SCHOLAR_LENS_MODAL_TOKEN`.
- In Modal, create a secret named `scholar-lens-api` with `SCHOLAR_LENS_MODAL_TOKEN`.

Optional Modal configuration:

```text
SCHOLAR_LENS_MODEL=nvidia/Llama-3.1-Nemotron-Nano-8B-v1
SCHOLAR_LENS_GPU=L4
```

Deploy with:

```bash
modal deploy modal_inference.py
```

After deployment, set these Gradio/Hugging Face Space environment variables to the URLs Modal prints:

```text
MODAL_SUMMARIZE_URL=https://your-summarize-endpoint.modal.run
MODAL_SYNTHESIZE_URL=https://your-synthesize-endpoint.modal.run
SCHOLAR_LENS_MODAL_TOKEN=your-shared-secret-token
SCHOLAR_LENS_CONTACT_EMAIL=you@example.com
```

The Modal endpoints reject requests that do not include the matching `Authorization: Bearer <token>` header.

Input protection:

- Summarization text is capped before it reaches Modal. Longer valid inputs are summarized in bounded chunks, then combined into a final summary.
- Grounded synthesis uses a fixed context budget so retrieved abstracts cannot overflow the model context window.
- The app also applies textbox limits in the UI and shows a friendly message when input is too large.
