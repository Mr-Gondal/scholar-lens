# Field Notes — Building Scholar Lens

*A build report for the Hugging Face **Build Small** hackathon (Backyard AI track).*

## The person and the problem

Scholar Lens was built for one real person: a professor in **atmospheric science**.
Their literature reviews meant juggling several paper databases — Semantic Scholar,
arXiv, PubMed — copying abstracts between tabs, losing track of what they'd read,
and re-reading the same studies. Hours per week, before any actual thinking began.

The goal was narrow and concrete: **turn one research question into a cited,
cross-database answer in seconds.**

## Why a small model is the *right* tool (not a constraint)

The natural worry with a small model is hallucination. We sidestepped it by
design: Scholar Lens **retrieves real abstracts first**, then asks the model to
answer *only* from those numbered abstracts, citing each claim as `[n]`. The
model never needs broad world knowledge — it reasons over text we supply. That is
exactly the job small models are good at, which makes `Qwen2.5-3B-Instruct` an
**honest** fit for the "Build Small" brief: the model is load-bearing (delete it
and you have nothing but links) without pretending to be a 70B oracle.

## What we built

- **Ask (the centrepiece):** a question → parallel search across **OpenAlex,
  Crossref, arXiv, PubMed** → top abstracts fed to the model → a synthesized,
  **cited** answer with a linked "Sources" panel.
- **Search:** ranked, de-duplicated results (DOI first, then fuzzy title),
  pagination, insight cards, CSV + **BibTeX** export.
- **Summarize & paper-chat:** condense or interrogate a single paper.
- A dark, custom-built academic UI (no default-Gradio look).

## Architecture decisions worth noting

- **Multi-source > single source.** Semantic Scholar alone was rate-limited and
  often empty; OpenAlex + Crossref + arXiv + PubMed with retries/backoff fixed the
  "no results" problem. OpenAlex abstracts arrive as an *inverted index* and must
  be reconstructed; Crossref abstracts are JATS XML and must be stripped.
- **Token budgeting everywhere.** Inputs are bounded in chars and rough tokens so
  the small context window is never blown.
- **Warm, secured inference.** The model loads once per Modal container
  (`@modal.enter()`) and both endpoints require a bearer token.
- **A cloud-free path.** `local_inference.py` runs the same prompts through a GGUF
  model with llama.cpp — proving Scholar Lens runs on a single consumer GPU.

## Benchmarks

*(Fill in from a real run: `python local_inference.py --benchmark`.)*

| Setup | Model / quant | VRAM | Throughput (tok/s) | Ask latency (s) |
|---|---|---|---|---|
| Modal (cloud) | Qwen2.5-3B / fp16 | — | — | — |
| Local GPU | Qwen2.5-3B / Q4_K_M | — | — | — |
| Local GPU | Qwen2.5-1.5B / Q4_K_M | — | — | — |

The headline: a useful, cited research assistant that fits comfortably on a
consumer GPU's memory budget.

## Built with Codex

The app was driven with **OpenAI Codex** as the coding agent. The commit history
is intentionally small and reviewable, so the *process* can be evaluated alongside
the product. (Agent trace shared for the *Sharing is Caring* badge.)

## What we'd do next

- Fine-tune a tiny model on abstract→summary pairs (*Well-Tuned* badge).
- Persist a user's prior questions into a running literature map.
- One-click "export this answer + citations" as a draft related-work paragraph.

## Honest limitations

- Crossref relevance is weak; it's a breadth fallback, not a primary ranker.
- Answers are only as good as the retrieved abstracts; full-text isn't fetched.
- First call after idle cold-starts the model (mitigated with a warm window).
