# Scholar Lens Professional Review

Reviewed on: 2026-06-13  
Reviewed files: `scholar-lens/app.py`, `scholar-lens/modal_inference.py`  
Also checked for consistency: `scholar-lens/README.md`, `scholar-lens/requirements.txt`

## Executive Summary

Scholar Lens has a solid core idea: it searches multiple academic sources, deduplicates results, and uses a small open model through Modal for summaries and grounded answers. That is a strong hackathon/product concept.

The main weaknesses are not that the app is "bad"; they are that the app feels unfinished in several places. Some UI affordances suggest behavior that does not actually work, the Modal model details are inconsistent, the public Modal endpoints are hardcoded, and there is no real protection against cost abuse, very large inputs, or degraded third-party API behavior.

If this were being judged professionally, I would prioritize fixing interaction polish, trust signals, reliability, and deployment hygiene before adding more visual decoration.

## What Is Already Good

- The concept is useful and clear: one research question can search OpenAlex, Crossref, arXiv, and PubMed.
- The app has a real "grounded answer" flow rather than only a generic chatbot.
- API searching is parallelized with `ThreadPoolExecutor`, so the app should feel faster than sequential search.
- The code escapes paper titles, authors, URLs, and source labels before rendering HTML, which is good for safety.
- Modal keeps the model loaded with `@modal.enter()`, avoiding repeated model loading for every request.
- `python -m py_compile app.py modal_inference.py` passes.
- `import app; app.build_app()` passes, so the Gradio component tree builds successfully.

## Highest Priority Issues

### 1. The visible search table looks clickable, but row click selection is not actually wired

Severity: High  
Location: `app.py:321`, `app.py:801`, `app.py:867-888`, `app.py:989-997`

The table rows use `cursor: pointer` and hover styling, which tells users they can click a result. There is also a `HEAD_SCRIPT` with a `selectPaper(index)` function, but the rendered rows do not include `onclick`, row IDs, radio IDs, or the hidden components that script expects.

Current reality: users must use the dropdown under the table, not the table itself.

Why this matters:

- Users will naturally click a result row and think the app is broken.
- The comment says click-to-select is powered by JavaScript, but the actual implementation moved to dropdown selection.
- This creates a trust problem in the most important workflow.

Recommendation:

- Either remove the fake clickable affordance and unused `HEAD_SCRIPT`, or fully wire row click selection.
- Best user experience: make each table row include a real "Summarize" button or a working row click action.
- If keeping the dropdown, rename it to "Choose from results" and make the table cursor default instead of pointer.

### 2. App says Qwen2.5-7B, Modal actually runs Qwen2.5-3B

Severity: High  
Location: `modal_inference.py:5`, `app.py:935`, `app.py:1067`, `README.md:15`, `README.md:60`

The deployed Modal file uses:

```python
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
```

But the app and README repeatedly say Qwen2.5-7B.

Why this matters:

- Judges or users may see this as inaccurate.
- If the project is for a model-size-limited challenge, model identity matters.
- It makes debugging harder because everyone thinks a different model is running.

Recommendation:

- Decide whether you want 3B or 7B.
- Update all text to match the actual model.
- If you keep 3B, present it confidently as a faster, cheaper small-model choice.
- If you move to 7B, update `MODEL_NAME` and confirm the GPU/cost tradeoff.

### 3. Modal endpoints are hardcoded and public

Severity: High  
Location: `app.py:20-21`, `modal_inference.py:73`, `modal_inference.py:90`

The app contains fixed public Modal URLs:

```python
MODAL_SUMMARIZE_URL = "https://..."
MODAL_SYNTHESIZE_URL = "https://..."
```

The Modal endpoints accept POST requests with no authentication, rate limiting, or request-size limits.

Why this matters:

- Anyone who finds the endpoint can call it directly.
- That can create unexpected Modal/GPU cost.
- A malicious or accidental huge request could waste compute or crash the container.

Recommendation:

- Load endpoint URLs from environment variables with safe local defaults.
- Add a simple shared secret header between the Gradio app and Modal endpoint.
- Add maximum input size checks before tokenization.
- Return proper HTTP error codes for bad requests instead of always returning a normal JSON body with `"error"`.

### 4. No input length protection before model tokenization

Severity: High  
Location: `modal_inference.py:51-67`, `modal_inference.py:73-89`, `modal_inference.py:90-119`, `app.py:460-485`, `app.py:528-551`

The app can send arbitrary text to Modal. The Modal endpoint then tokenizes the whole prompt. There is no hard character limit, token limit, chunking strategy, or graceful message for oversized input.

Why this matters:

- A pasted PDF or long paper can exceed the model context.
- Tokenization and generation can become slow or fail.
- This is a common production failure point for LLM apps.

Recommendation:

- Add frontend limits in Gradio and backend limits in Modal.
- For summarization, chunk long text and summarize chunks before a final summary.
- For synthesis, continue using abstracts, but enforce a token budget, not only a character budget.

### 5. Clear button is incomplete

Severity: Medium  
Location: `app.py:637-648`, `app.py:1084-1096`

The Clear button resets search status, results, paper state, page label, selected paper, source text, and summary output. It does not clear:

- The search input textbox.
- The Ask question textbox.
- The Ask answer output.
- The references output.
- The summarize load status message.

Why this matters:

- After clicking Clear, old content can remain on screen.
- Users may think the app state is reset when it is only partly reset.

Recommendation:

- Make Clear reset all visible state connected to the Search and Summarize workflow.
- Consider separate "Clear Search" and "Clear All" buttons if you do not want to wipe the Ask tab.

## Functional and UX Gaps

### Search result selection is awkward

Severity: Medium  
Location: `app.py:988-997`

Using a dropdown to choose one paper from many long paper titles is not ideal. Long titles can be hard to scan in a dropdown, and it separates selection from the result row the user just read.

Recommendation:

- Add an action column with "Summarize" or "Use in Ask" per row.
- Show the selected result inline.
- Add an abstract preview drawer or expandable row.

### Pagination buttons are always active

Severity: Medium  
Location: `app.py:456-457`, `app.py:983-986`, `app.py:1020-1028`

Prev and Next clamp the page internally, so they will not crash. But they are still clickable when there are no results, or when the user is already on the first/last page.

Recommendation:

- Return `gr.update(interactive=False)` for Prev/Next when unavailable.
- Or replace buttons with a compact page selector.

### The app sorts mostly by year, not relevance

Severity: Medium  
Location: `app.py:407-412`

After collecting results, the app sorts by year descending. This can push recent but less relevant papers above older, more important papers.

Recommendation:

- Preserve each API's relevance ordering and add year as secondary metadata.
- Or compute a simple relevance score based on title/abstract query matches, citations, and source rank.

### Ask mode uses the whole user question as the search query

Severity: Medium  
Location: `app.py:554-578`

The Ask tab sends the natural-language question directly to academic search APIs. That sometimes works, but academic APIs often perform better with extracted keywords.

Recommendation:

- Add a lightweight query-refinement step before search.
- For example, extract 3-8 academic keywords from the question.
- Show "Search terms used" so the user trusts the answer pipeline.

### Mobile/table responsiveness is weak

Severity: Medium  
Location: `app.py:771-789`

The results table is wide, but `.table-shell` uses `overflow: hidden`. On smaller screens, columns may clip or become unpleasant to use.

Recommendation:

- Use `overflow-x: auto`.
- On mobile, hide citations/authors or convert rows into compact result cards.
- Keep title, year, source, and primary action visible.

### Empty labels hurt accessibility

Severity: Low to Medium  
Location: `app.py:940-945`, `app.py:970-975`

The main textboxes use `label=""`. This can make the UI cleaner visually, but it is weaker for screen readers and form clarity.

Recommendation:

- Use meaningful labels like "Research question" and "Search topic".
- If you want a clean look, style labels subtly instead of removing them.

### Some CSS classes are declared in components but not styled

Severity: Low  
Location: `app.py:977`, `app.py:980`, `app.py:988`, `app.py:1032`, `app.py:1057`

Classes such as `btn-crimson`, `status-line`, `action-row`, `summarize-panel`, and `about-card` appear in the UI, but there are no matching CSS rules for several of them.

Recommendation:

- Either style them or remove the class names.
- The About tab especially could use stronger visual treatment if this is for a demo.

## Reliability and Data Quality Gaps

### No retry or backoff for source APIs

Severity: Medium  
Location: `app.py:62-73`, `app.py:388-413`

The app calls external APIs once and fails quickly. Academic APIs sometimes rate limit or temporarily fail.

Recommendation:

- Use a `requests.Session`.
- Add small retry/backoff for 429, 500, 502, 503, and 504.
- Log the real exception server-side while showing a clean user message.

### Source-specific query quality can be improved

Severity: Medium  
Location: `app.py:102-287`

Each source has different search behavior. The current implementation uses one simple query strategy for all sources.

Recommendation:

- For PubMed, support MeSH-like biomedical terms where possible.
- For arXiv, consider searching title/abstract fields more intentionally.
- For Crossref/OpenAlex, request fields that help ranking and display.

### Dedupe is useful but basic

Severity: Low to Medium  
Location: `app.py:364-385`

The app deduplicates by DOI and heavily normalized title. That is fine for a first pass, but it will miss near-duplicates and can occasionally merge titles that should remain separate.

Recommendation:

- Keep DOI as primary.
- Add fuzzy title matching with a threshold.
- Prefer the result with the best metadata/abstract when duplicates are found.

## Modal/LLM Quality Gaps

### Generation is too random for research citation work

Severity: Medium  
Location: `modal_inference.py:64-66`

The model uses:

```python
temperature=0.7
do_sample=True
```

For grounded academic answers, this may produce more varied but less stable answers.

Recommendation:

- Use lower temperature, for example `0.2`.
- Consider `do_sample=False` for more deterministic citation behavior.
- Keep summaries concise and structured.

### Modal GPU choice may be overkill for a 3B 4-bit model

Severity: Medium  
Location: `modal_inference.py:21`, `modal_inference.py:35-47`

The code requests `gpu="A100"` while running a 3B model in 4-bit. That is likely more expensive than needed.

Recommendation:

- Test a cheaper GPU class for the 3B model.
- Keep A100 only if latency or memory actually requires it.
- If using 7B, still benchmark cost/performance.

### Error responses expose internal exception text

Severity: Medium  
Location: `modal_inference.py:86-88`, `modal_inference.py:117-119`

Returning raw exception text can leak implementation details.

Recommendation:

- Return a generic user-facing message.
- Log the real exception inside Modal.
- Include a request ID if you want easier debugging.

## Presentation and "Attraction" Recommendations

These are the changes most likely to make the app feel more polished to users or judges.

### Add result insight cards above the table

Show quick stats after search:

- Total papers found.
- Sources represented.
- Year range.
- How many have abstracts.
- Top source by result count.

This gives users instant confidence that the app did real work.

### Add sample prompt chips on the Ask tab

Add 3-5 clickable example questions. This reduces blank-page friction and makes demos smoother.

Examples:

- "How is AI being used for early cancer detection?"
- "What are recent methods for battery degradation prediction?"
- "Where do papers disagree on retrieval-augmented generation?"

### Add a visible evidence panel

The app already shows references, but you can make it more compelling:

- Show each cited paper with title, year, source, and abstract snippet.
- Highlight which papers were used in the answer.
- Add "Open source" links beside each citation.

### Add export/copy actions

Very useful additions:

- Copy answer.
- Download references as CSV.
- Export selected paper summary as Markdown.
- Copy citation list.

These are small features that make the app feel practical rather than just a demo.

### Add loading messages that explain what is happening

When Ask runs, the app may take time. Show staged progress:

- Searching OpenAlex, Crossref, arXiv, PubMed.
- Filtering papers with abstracts.
- Building grounded context.
- Asking the model.

This reduces user impatience during Modal cold starts.

## Code Structure Recommendations

### Split the 1100-line `app.py`

Severity: Medium  
Location: `app.py`

The app currently mixes:

- API clients.
- Data models.
- Search aggregation.
- HTML rendering.
- CSS.
- JavaScript.
- Gradio wiring.
- Modal client calls.

Recommendation:

Create a structure like:

```text
scholar_lens/
  models.py
  sources.py
  ranking.py
  modal_client.py
  render.py
  ui.py
app.py
modal_inference.py
```

This will make the project easier to test and improve.

### Add tests

Severity: Medium

There are no tests. For a project like this, a few focused tests would add a lot of confidence.

Recommended tests:

- `_reconstruct_abstract`
- `_normalize_doi`
- `_dedupe_results`
- `_build_synthesis_context`
- `load_selected_paper`
- mocked search functions for each source
- mocked Modal client behavior for timeout/error/success

## Suggested Fix Order

1. Fix model identity mismatch: 3B vs 7B.
2. Fix search result selection UX: either working row click or no fake row click.
3. Move Modal URLs and contact email into environment variables.
4. Add Modal request authentication and input size limits.
5. Improve Clear button behavior.
6. Add mobile-friendly result layout.
7. Add search insight cards and sample prompt chips.
8. Add retry/backoff for external APIs.
9. Lower generation randomness for grounded academic answers.
10. Add tests for core data functions.

## Professional Verdict

Scholar Lens has a good foundation and a relevant use case. The project is not lacking in idea; it is lacking in finish, trust, and operational safety.

The most damaging issue from a user's perspective is the misleading clickable table. The most damaging issue from a deployment perspective is the public, hardcoded Modal endpoint with no input limits. The most damaging issue from a credibility perspective is the mismatch between the advertised model and the actual model.

Fix those first. Then add polish features like insight cards, sample questions, export buttons, and a better evidence panel. Those will make the app feel much more complete without changing the core architecture.

