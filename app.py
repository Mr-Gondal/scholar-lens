from __future__ import annotations

import html
import re
import textwrap
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import gradio as gr
import requests

APP_TITLE = "Scholar Lens"
SEARCH_LIMIT_PER_SOURCE = 8
REQUEST_TIMEOUT_SECONDS = 12
MODAL_SUMMARIZE_URL = "https://kinggondal731--scholar-lens-summarizer-summarize-paper.modal.run"


@dataclass(frozen=True)
class PaperResult:
    title: str
    year: str
    source: str
    authors: str
    citations: str
    url: str
    abstract: str = ""


def _safe_text(value: Any, fallback: str = "Unknown") -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _shorten_authors(authors: list[str], max_authors: int = 3) -> str:
    clean_authors = [author.strip() for author in authors if author and author.strip()]
    if not clean_authors:
        return "Unknown"
    if len(clean_authors) <= max_authors:
        return ", ".join(clean_authors)
    return f"{', '.join(clean_authors[:max_authors])}, et al."


def _extract_year(date_value: Any) -> str:
    text = _safe_text(date_value, "")
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else "Unknown"


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
            "Accept": "application/json",
        },
    )
    response.raise_for_status()
    return response.json()


def search_semantic_scholar(query: str) -> tuple[list[PaperResult], str | None]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": SEARCH_LIMIT_PER_SOURCE,
        "fields": "title,year,authors,citationCount,url,abstract",
    }
    try:
        payload = _request_json(url, params)
    except requests.RequestException:
        return [], "Semantic Scholar is unavailable right now."

    results: list[PaperResult] = []
    for paper in payload.get("data", []):
        authors = [author.get("name", "") for author in paper.get("authors", [])]
        results.append(
            PaperResult(
                title=_safe_text(paper.get("title"), "Untitled paper"),
                year=_safe_text(paper.get("year")),
                source="Semantic Scholar",
                authors=_shorten_authors(authors),
                citations=_safe_text(paper.get("citationCount"), "0"),
                url=_safe_text(paper.get("url"), "#"),
                abstract=_safe_text(paper.get("abstract"), ""),
            )
        )
    return results, None


def search_arxiv(query: str) -> tuple[list[PaperResult], str | None]:
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": SEARCH_LIMIT_PER_SOURCE,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except (requests.RequestException, ET.ParseError):
        return [], "arXiv is unavailable right now."

    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    results: list[PaperResult] = []
    for entry in root.findall("atom:entry", namespace):
        title = " ".join((entry.findtext("atom:title", default="", namespaces=namespace)).split())
        published = entry.findtext("atom:published", default="", namespaces=namespace)
        link = entry.findtext("atom:id", default="#", namespaces=namespace)
        abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=namespace)).split())
        authors = [
            author.findtext("atom:name", default="", namespaces=namespace)
            for author in entry.findall("atom:author", namespace)
        ]
        results.append(
            PaperResult(
                title=_safe_text(title, "Untitled paper"),
                year=_extract_year(published),
                source="arXiv",
                authors=_shorten_authors(authors),
                citations="N/A",
                url=_safe_text(link, "#"),
                abstract=_safe_text(abstract, ""),
            )
        )
    return results, None


def search_pubmed(query: str) -> tuple[list[PaperResult], str | None]:
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": SEARCH_LIMIT_PER_SOURCE,
        "sort": "relevance",
    }
    try:
        search_payload = _request_json(search_url, search_params)
        paper_ids = search_payload.get("esearchresult", {}).get("idlist", [])
        if not paper_ids:
            return [], None
        summary_payload = _request_json(
            summary_url,
            {"db": "pubmed", "id": ",".join(paper_ids), "retmode": "json"},
        )
        abstracts_by_id = _fetch_pubmed_abstracts(paper_ids)
    except (requests.RequestException, ET.ParseError):
        return [], "PubMed is unavailable right now."

    summaries = summary_payload.get("result", {})
    results: list[PaperResult] = []
    for paper_id in paper_ids:
        item = summaries.get(paper_id, {})
        authors = [author.get("name", "") for author in item.get("authors", [])]
        results.append(
            PaperResult(
                title=_safe_text(item.get("title"), "Untitled paper"),
                year=_extract_year(item.get("pubdate")),
                source="PubMed",
                authors=_shorten_authors(authors),
                citations="N/A",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
                abstract=abstracts_by_id.get(paper_id, ""),
            )
        )
    return results, None


def _fetch_pubmed_abstracts(paper_ids: list[str]) -> dict[str, str]:
    if not paper_ids:
        return {}
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    response = requests.get(
        fetch_url,
        params={
            "db": "pubmed",
            "id": ",".join(paper_ids),
            "retmode": "xml",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    abstracts: dict[str, str] = {}
    for article in root.findall(".//PubmedArticle"):
        paper_id = article.findtext(".//PMID", default="")
        abstract_parts = [
            " ".join(part.itertext()).strip()
            for part in article.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(part for part in abstract_parts if part)
        if paper_id and abstract:
            abstracts[paper_id] = abstract
    return abstracts


def _source_badge(source: str) -> str:
    config = {
        "Semantic Scholar": ("semantic", "🧠"),
        "arXiv": ("arxiv", "📐"),
        "PubMed": ("pubmed", "🧬"),
    }
    class_name, icon = config.get(source, ("default", "📄"))
    return (
        f'<span class="source-badge {class_name}">'
        f'<span class="badge-icon">{icon}</span>{html.escape(source)}'
        f'</span>'
    )


def _render_results_table(results: list[PaperResult]) -> str:
    if not results:
        return """
        <div class="empty-state">
            <div class="empty-crest">📚</div>
            <h3>Your library awaits</h3>
            <p>Run a search to populate your scholarly results below.</p>
        </div>
        """
    rows = []
    for i, result in enumerate(results):
        safe_url = html.escape(result.url, quote=True)
        rows.append(
            textwrap.dedent(
                f"""
                <tr class="result-row" onclick="selectPaper({i})" id="row-{i}">
                    <td class="select-cell">
                        <input type="radio" name="paper-select" id="radio-{i}" onclick="event.stopPropagation(); selectPaper({i})">
                    </td>
                    <td class="title-cell">{html.escape(result.title)}</td>
                    <td class="year-cell">{html.escape(result.year)}</td>
                    <td>{_source_badge(result.source)}</td>
                    <td class="authors-cell">{html.escape(result.authors)}</td>
                    <td class="citation-cell">{html.escape(result.citations)}</td>
                    <td><a class="paper-link" href="{safe_url}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()">Open ↗</a></td>
                </tr>
                """
            ).strip()
        )
    return f"""
    <div class="table-shell">
        <table class="results-table">
            <thead>
                <tr>
                    <th style="width: 40px;"></th>
                    <th>Title</th>
                    <th>Year</th>
                    <th>Source</th>
                    <th>Authors</th>
                    <th>Cites</th>
                    <th>Link</th>
                </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """


def _dedupe_results(results: list[PaperResult]) -> list[PaperResult]:
    """Drop duplicate papers that appear in more than one source.

    Papers are matched on a normalized title; the first occurrence wins so we
    keep the source order produced by the search functions.
    """
    seen: set[str] = set()
    unique: list[PaperResult] = []
    for result in results:
        key = re.sub(r"[^a-z0-9]+", "", result.title.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def search_all_sources(query: str) -> tuple[str, str, list[PaperResult], int | None]:
    clean_query = query.strip()
    if not clean_query:
        return (
            "Enter a research topic to search Semantic Scholar, arXiv, and PubMed.",
            _render_results_table([]),
            [],
            None,
        )

    search_functions = [search_semantic_scholar, search_arxiv, search_pubmed]
    results: list[PaperResult] = []
    warnings: list[str] = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_search = {executor.submit(fn, clean_query): fn.__name__ for fn in search_functions}
        for future in future_to_search:
            try:
                source_results, warning = future.result()
                results.extend(source_results)
                if warning:
                    warnings.append(warning)
            except Exception as e:
                warnings.append(f"An error occurred searching {future_to_search[future]}: {str(e)}")

    results = _dedupe_results(results)

    def _sort_key(item: PaperResult) -> tuple[int, int]:
        if item.year == "Unknown" or not item.year.isdigit():
            return (0, 0)
        return (1, int(item.year))

    results.sort(key=_sort_key, reverse=True)

    if warnings and results:
        status = f"✓ Found **{len(results)}** papers. " + " ".join(warnings)
    elif warnings:
        status = " ".join(warnings)
    else:
        status = f"✓ Found **{len(results)}** papers across all sources."

    if not results:
        status = "No papers found. Try a broader research topic or a different phrase."

    return (
        status,
        _render_results_table(results),
        results,
        None,
    )


def summarize_with_modal(text: str) -> str:
    if not text or len(text.strip()) < 50:
        return "Please provide a longer abstract or paper text to summarize."
    try:
        response = requests.post(
            MODAL_SUMMARIZE_URL,
            json={"text": text},
            timeout=120,
        )
        response.raise_for_status()
    except requests.Timeout:
        return (
            "The AI summarizer timed out (the model may be cold-starting). "
            "Please try again in a few seconds."
        )
    except requests.RequestException:
        return "The AI summarizer is unavailable right now. Please try again shortly."

    try:
        payload = response.json()
    except ValueError:
        return "The AI summarizer returned an unexpected response. Please try again shortly."

    if isinstance(payload, dict):
        return _safe_text(payload.get("summary") or payload.get("error"), "No summary returned.")
    return _safe_text(payload, "No summary returned.")


def select_result(event: gr.SelectData) -> int | None:
    if event.index is None:
        return None
    index = event.index
    if isinstance(index, (list, tuple)):
        if not index:
            return None
        index = index[0]
    return int(index)


def load_selected_paper(
    selected_index: int | None,
    results: list[PaperResult],
) -> tuple[str, str, str, gr.Tabs]:
    if selected_index is None or selected_index >= len(results):
        return (
            "",
            "Select a paper from the results table first.",
            "",
            gr.update(selected="summarize"),
        )
    paper = results[selected_index]
    abstract = paper.abstract.strip()
    if not abstract:
        return (
            f"{paper.title}\n\nNo abstract is available for this result. Paste paper text here to summarize it.",
            "This paper does not include an abstract from the source API.",
            "",
            gr.update(selected="summarize"),
        )
    paper_text = f"Title: {paper.title}\n\nAbstract: {abstract}"
    return paper_text, f"📖 Loaded: *{paper.title}*", "", gr.update(selected="summarize")


def summarize_selected_paper(
    selected_index: int | None,
    results: list[PaperResult],
) -> tuple[str, str, str, gr.Tabs]:
    paper_text, load_status, _, tab_update = load_selected_paper(selected_index, results)
    if not paper_text:
        return paper_text, load_status, "", tab_update
    if "No abstract is available" in paper_text:
        return (
            paper_text,
            load_status,
            "No abstract was available to summarize. Paste the paper text above and click Summarize with AI.",
            tab_update,
        )
    return paper_text, load_status, summarize_with_modal(paper_text), tab_update


def clear_search() -> tuple[str, str, list[PaperResult], int | None, str, str]:
    return (
        "Enter a research topic to begin.",
        _render_results_table([]),
        [],
        None,
        "",
        "",
    )


# Professional dark theme: slate navy + electric blue + clean research typography
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

:root {
    --sl-bg: #0f172a;
    --sl-bg-deep: #020617;
    --sl-panel: #1e293b;
    --sl-panel-soft: #334155;
    --sl-border: #334155;
    --sl-border-soft: #1e293b;
    --sl-text: #f8fafc;
    --sl-text-bright: #ffffff;
    --sl-muted: #94a3b8;
    --sl-accent: #3b82f6;
    --sl-accent-bright: #60a5fa;
    --sl-accent-soft: rgba(59, 130, 246, 0.15);
    --sl-ink: #1e293b;
    --sl-crimson: #ef4444;
    --sl-emerald: #10b981;
}

/* ===== GLOBAL CANVAS ===== */
.gradio-container {
    background:
        radial-gradient(1200px 600px at 15% -10%, rgba(59, 130, 246, 0.08), transparent 60%),
        var(--sl-bg) !important;
    color: var(--sl-text) !important;
    font-family: "Inter", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    min-height: 100vh;
}

.main-shell { max-width: 1200px; margin: 0 auto; padding: 0 8px; }

/* ===== ACADEMIC HEADER ===== */
.app-header {
    position: relative;
    padding: 36px 28px 28px;
    margin-bottom: 22px;
    border-radius: 14px;
    background: var(--sl-panel);
    border: 1px solid var(--sl-border);
    border-top: 4px solid var(--sl-accent);
    overflow: hidden;
}

.header-row { display: flex; align-items: center; gap: 20px; position: relative; }
.crest {
    width: 64px; height: 64px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--sl-accent-bright), var(--sl-accent));
    display: flex; align-items: center; justify-content: center;
    font-size: 30px;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.35);
    flex-shrink: 0;
}
.header-text h1 {
    margin: 0;
    color: var(--sl-text-bright);
    font-family: "Playfair Display", "Georgia", serif;
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.header-text h1 .accent { color: var(--sl-accent-bright); }
.header-text p {
    margin: 6px 0 0;
    color: var(--sl-muted);
    font-size: 14px;
}
.header-chips {
    display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;
}
.header-chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 11px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 600;
    background: var(--sl-accent-soft);
    color: var(--sl-accent-bright);
    border: 1px solid rgba(59, 130, 246, 0.25);
}

/* ===== TABS ===== */
.tab-nav { border-bottom: 1px solid var(--sl-border) !important; }
.tab-nav button.selected {
    color: var(--sl-accent-bright) !important;
    border-bottom: 2px solid var(--sl-accent) !important;
    background: var(--sl-accent-soft) !important;
}

/* ===== INPUTS ===== */
.gradio-container input[type=text],
.gradio-container input[type=search],
.gradio-container textarea {
    background: var(--sl-bg-deep) !important;
    border: 1px solid var(--sl-border) !important;
    color: var(--sl-text-bright) !important;
    border-radius: 8px !important;
}
.gradio-container input[type=text]:focus,
.gradio-container textarea:focus {
    border-color: var(--sl-accent) !important;
    box-shadow: 0 0 0 3px var(--sl-accent-soft) !important;
}

/* ===== BUTTONS ===== */
.gradio-container button.primary {
    background: var(--sl-accent) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.30) !important;
}
.gradio-container button.primary:hover {
    background: var(--sl-accent-bright) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.45) !important;
}

/* ===== RESULTS TABLE ===== */
.table-shell {
    border: 1px solid var(--sl-border);
    border-radius: 12px;
    background: var(--sl-panel);
    overflow: hidden;
}
.results-table { width: 100%; border-collapse: collapse; }
.results-table th {
    background: var(--sl-bg-deep);
    color: var(--sl-accent-bright) !important;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    padding: 14px 18px;
}
.results-table td {
    padding: 16px 18px;
    border-bottom: 1px solid var(--sl-border-soft);
}
.select-cell {
    width: 40px;
    text-align: center;
}
.select-cell input[type="radio"] {
    accent-color: var(--sl-accent);
    cursor: pointer;
    width: 18px;
    height: 18px;
}
.result-row { cursor: pointer; transition: background 140ms ease; }
.result-row:hover { background: var(--sl-panel-soft); }
.result-row.selected {
    background: var(--sl-accent-soft) !important;
    border-left: 4px solid var(--sl-accent);
}

.title-cell {
    font-family: "Playfair Display", serif;
    font-weight: 600;
    color: var(--sl-text-bright) !important;
}

.source-badge.semantic { color: #60a5fa !important; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); }
.source-badge.arxiv { color: #f87171 !important; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); }
.source-badge.pubmed { color: #34d399 !important; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); }

/* Remove system buttons and footers */
.settings, footer, .show-api { display: none !important; }

.hidden-component { display: none !important; }
"""

def build_app() -> tuple[gr.Blocks, gr.themes.Base]:
    theme = gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    ).set(
        body_background_fill="#0f172a",
        body_text_color="#f8fafc",
        block_background_fill="#1e293b",
        block_border_color="#334155",
        button_primary_background_fill="#3b82f6",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(title=APP_TITLE, head="""
        <script>
        function selectPaper(index) {
            // Update the hidden input
            const input = document.querySelector('#hidden_index_input textarea');
            if (input) {
                input.value = index;
                input.dispatchEvent(new Event('input', { bubbles: true }));
            }
            
            // Highlight the row
            document.querySelectorAll('.result-row').forEach(r => r.classList.remove('selected'));
            const row = document.getElementById('row-' + index);
            if (row) row.classList.add('selected');
            
            // Check the radio button
            const radio = document.getElementById('radio-' + index);
            if (radio) radio.checked = true;
            
            // Trigger the hidden button
            setTimeout(() => {
                const btn = document.querySelector('#hidden_select_btn button');
                if (btn) btn.click();
            }, 50);
        }
        </script>
    """) as app:
        with gr.Column(elem_classes=["main-shell"]):
            gr.HTML(
                """
                <header class="app-header">
                    <div class="header-row">
                        <div class="crest">🔬</div>
                        <div class="header-text">
                            <h1>Scholar <span class="accent">Lens</span></h1>
                            <p>Professional Research Discovery & Synthesis Engine</p>
                            <div class="header-chips">
                                <span class="header-chip">Semantic Scholar</span>
                                <span class="header-chip">arXiv</span>
                                <span class="header-chip">PubMed</span>
                            </div>
                        </div>
                    </div>
                </header>
                """
            )

            papers_state = gr.State([])
            selected_index_state = gr.State(None)
            
            # Hidden components for JS communication
            hidden_index_input = gr.Textbox(visible=False, elem_id="hidden_index_input")
            hidden_select_btn = gr.Button(visible=False, elem_id="hidden_select_btn")

            with gr.Tabs(selected="search") as app_tabs:
                with gr.Tab("🔍  Search", id="search"):
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="",
                            placeholder="Enter research topic (e.g., 'Quantum computing in drug discovery')",
                            scale=5,
                            container=False,
                        )
                        search_button = gr.Button("🔍 Search", variant="primary", scale=1)
                        with gr.Column(scale=0, min_width=100, elem_classes=["btn-crimson"]):
                            clear_button = gr.Button("Clear")

                    status_output = gr.Markdown("Ready for search.", elem_classes=["status-line"])
                    results_output = gr.HTML(_render_results_table([]))

                    with gr.Row(elem_classes=["action-row"]):
                        with gr.Column(scale=1):
                            summarize_selected_button = gr.Button("✨ Summarize Now", variant="primary")

                    search_button.click(
                        fn=search_all_sources,
                        inputs=query_input,
                        outputs=[status_output, results_output, papers_state, selected_index_state],
                        show_progress="full",
                    )
                    query_input.submit(
                        fn=search_all_sources,
                        inputs=query_input,
                        outputs=[status_output, results_output, papers_state, selected_index_state],
                        show_progress="full",
                    )
                    
                    # JS row selection logic
                    hidden_select_btn.click(
                        fn=lambda x: int(x) if x else None,
                        inputs=hidden_index_input,
                        outputs=selected_index_state
                    )

                with gr.Tab("✨  Summarize", id="summarize"):
                    with gr.Column(elem_classes=["summarize-panel"]):
                        load_status_output = gr.Markdown("Select a paper to summarize.")
                        source_text = gr.Textbox(label="Paper Context", lines=10)
                        summarize_button = gr.Button("✨ Summarize with AI", variant="primary")
                        summary_output = gr.Textbox(label="AI Synthesis", lines=8, interactive=False)

                    summarize_button.click(
                        fn=summarize_with_modal, 
                        inputs=source_text, 
                        outputs=summary_output,
                        show_progress="full",
                    )
                    
                    summarize_selected_button.click(
                        fn=summarize_selected_paper,
                        inputs=[selected_index_state, papers_state],
                        outputs=[source_text, load_status_output, summary_output, app_tabs],
                        show_progress="full",
                    )

                with gr.Tab("📖  About"):
                    gr.HTML(
                        """
                        <div class="about-card">
                            <h2>🎓 About Scholar Lens</h2>
                            <p>
                                <strong>Scholar Lens</strong> is a professional academic discovery engine
                                built to accelerate literature reviews and research synthesis.
                            </p>
                            <p>
                                It performs real-time searches across <strong>Semantic Scholar</strong>, 
                                <strong>arXiv</strong>, and <strong>PubMed</strong> using parallel processing, 
                                then distills findings using a high-performance LLM hosted on Modal.
                            </p>
                            <p style="margin-top: 20px; font-size: 13px; color: var(--sl-accent-bright);">
                                Background: #0f172a | Accent: #3b82f6 | V2.0
                            </p>
                        </div>
                        """
                    )
            
            clear_button.click(
                fn=clear_search,
                outputs=[status_output, results_output, papers_state, selected_index_state, source_text, summary_output],
            )

    return app, theme


if __name__ == "__main__":
    print("Starting app...")
    app, theme = build_app()
    print("App built, launching...")
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=theme,
        css=CUSTOM_CSS,
    )
