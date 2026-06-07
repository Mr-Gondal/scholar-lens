MODAL_SUMMARIZE_URL = "https://kinggondal731--scholar-lens-summarizer-summarize-paper.modal.run"
from __future__ import annotations

import html
import re
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import gradio as gr
import requests


APP_TITLE = "Scholar Lens"
SEARCH_LIMIT_PER_SOURCE = 8
REQUEST_TIMEOUT_SECONDS = 12


@dataclass(frozen=True)
class PaperResult:
    title: str
    year: str
    source: str
    authors: str
    citations: str
    url: str


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
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def search_semantic_scholar(query: str) -> tuple[list[PaperResult], str | None]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": SEARCH_LIMIT_PER_SOURCE,
        "fields": "title,year,authors,citationCount,url",
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
    except requests.RequestException:
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
            )
        )
    return results, None


def _source_badge(source: str) -> str:
    classes = {
        "Semantic Scholar": "semantic",
        "arXiv": "arxiv",
        "PubMed": "pubmed",
    }
    class_name = classes.get(source, "default")
    return f'<span class="source-badge {class_name}">{html.escape(source)}</span>'


def _render_results_table(results: list[PaperResult]) -> str:
    if not results:
        return """
        <div class="empty-state">
            <h3>No papers found</h3>
            <p>Try a broader research topic or a different phrase.</p>
        </div>
        """

    rows = []
    for result in results:
        safe_url = html.escape(result.url, quote=True)
        rows.append(
            textwrap.dedent(
                f"""
                <tr class="result-row" onclick="window.open('{safe_url}', '_blank', 'noopener,noreferrer')">
                    <td class="title-cell">{html.escape(result.title)}</td>
                    <td>{html.escape(result.year)}</td>
                    <td>{_source_badge(result.source)}</td>
                    <td>{html.escape(result.authors)}</td>
                    <td class="citation-cell">{html.escape(result.citations)}</td>
                </tr>
                """
            ).strip()
        )

    return f"""
    <div class="table-shell">
        <table class="results-table">
            <thead>
                <tr>
                    <th>Title</th>
                    <th>Year</th>
                    <th>Source</th>
                    <th>Authors</th>
                    <th>Citations</th>
                </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """


def search_all_sources(query: str) -> tuple[str, str]:
    clean_query = query.strip()
    if not clean_query:
        return (
            "Enter a research topic to search Semantic Scholar, arXiv, and PubMed.",
            _render_results_table([]),
        )

    search_functions = (search_semantic_scholar, search_arxiv, search_pubmed)
    results: list[PaperResult] = []
    warnings: list[str] = []

    for search_function in search_functions:
        source_results, warning = search_function(clean_query)
        results.extend(source_results)
        if warning:
            warnings.append(warning)

    results.sort(key=lambda item: (item.year == "Unknown", item.year), reverse=True)

    if warnings and results:
        status = f"Found {len(results)} papers. " + " ".join(warnings)
    elif warnings:
        status = " ".join(warnings)
    else:
        status = f"Found {len(results)} papers across all sources."

    return status, _render_results_table(results)


def summarize_with_modal(text: str) -> str:
    """Call Modal to generate a real AI summary"""
    if not text or len(text.strip()) < 50:
        return "Please provide a longer abstract or paper text to summarize."

    try:
        response = requests.post(
            MODAL_SUMMARIZE_URL,
            json={"text": text},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

CUSTOM_CSS = """
:root {
    --sl-bg: #0f172a;
    --sl-panel: #111c33;
    --sl-panel-soft: #17233d;
    --sl-border: #263551;
    --sl-text: #e5edf8;
    --sl-muted: #93a4bb;
    --sl-accent: #3b82f6;
}

.gradio-container {
    background: var(--sl-bg) !important;
    color: var(--sl-text) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.main-shell {
    max-width: 1180px;
    margin: 0 auto;
}

.app-header {
    padding: 28px 0 18px;
    border-bottom: 1px solid var(--sl-border);
    margin-bottom: 18px;
}

.app-header h1 {
    margin: 0;
    color: #f8fbff;
    font-size: 34px;
    line-height: 1.1;
    letter-spacing: 0;
}

.app-header p {
    margin: 8px 0 0;
    color: var(--sl-muted);
    font-size: 15px;
}

.gradio-container button.primary {
    background: var(--sl-accent) !important;
    border: 1px solid #60a5fa !important;
    color: white !important;
    font-weight: 700 !important;
}

.gradio-container input,
.gradio-container textarea {
    background: #0b1222 !important;
    border-color: var(--sl-border) !important;
    color: var(--sl-text) !important;
}

.tab-nav button {
    color: var(--sl-muted) !important;
}

.tab-nav button.selected {
    color: #ffffff !important;
    border-color: var(--sl-accent) !important;
}

.status-line {
    color: var(--sl-muted);
    min-height: 24px;
}

.table-shell {
    overflow-x: auto;
    border: 1px solid var(--sl-border);
    border-radius: 8px;
    background: var(--sl-panel);
}

.results-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 860px;
}

.results-table th {
    background: #0b1222;
    color: #c9d7ea;
    text-align: left;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0;
    padding: 13px 16px;
    border-bottom: 1px solid var(--sl-border);
}

.results-table td {
    color: var(--sl-text);
    padding: 15px 16px;
    border-bottom: 1px solid var(--sl-border);
    vertical-align: middle;
    font-size: 14px;
}

.result-row {
    cursor: pointer;
    transition: background 140ms ease, transform 140ms ease;
}

.result-row:hover {
    background: var(--sl-panel-soft);
}

.title-cell {
    max-width: 520px;
    font-weight: 650;
    line-height: 1.35;
}

.citation-cell {
    font-variant-numeric: tabular-nums;
}

.source-badge {
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}

.source-badge.semantic {
    color: #bfdbfe;
    background: rgba(59, 130, 246, 0.18);
    border: 1px solid rgba(96, 165, 250, 0.32);
}

.source-badge.arxiv {
    color: #fecaca;
    background: rgba(239, 68, 68, 0.17);
    border: 1px solid rgba(248, 113, 113, 0.30);
}

.source-badge.pubmed {
    color: #bbf7d0;
    background: rgba(34, 197, 94, 0.16);
    border: 1px solid rgba(74, 222, 128, 0.30);
}

.empty-state {
    border: 1px dashed var(--sl-border);
    border-radius: 8px;
    padding: 34px;
    text-align: center;
    background: rgba(17, 28, 51, 0.72);
}

.empty-state h3 {
    margin: 0 0 6px;
    color: #f8fbff;
}

.empty-state p {
    margin: 0;
    color: var(--sl-muted);
}
"""


def build_app() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    ).set(
        body_background_fill="#0f172a",
        body_text_color="#e5edf8",
        block_background_fill="#111c33",
        block_border_color="#263551",
        button_primary_background_fill="#3b82f6",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(title=APP_TITLE, theme=theme, css=CUSTOM_CSS) as app:
        with gr.Column(elem_classes=["main-shell"]):
            gr.HTML(
                """
                <header class="app-header">
                    <h1>Scholar Lens</h1>
                    <p>Search 200M+ papers across Semantic Scholar, arXiv &amp; PubMed</p>
                </header>
                """
            )

            with gr.Tabs():
                with gr.Tab("Search"):
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="",
                            placeholder="Search any research topic...",
                            scale=5,
                            container=False,
                        )
                        search_button = gr.Button(
                            "Search All Sources",
                            variant="primary",
                            scale=1,
                        )

                    status_output = gr.Markdown(
                        "Enter a research topic to begin.",
                        elem_classes=["status-line"],
                    )
                    results_output = gr.HTML(_render_results_table([]))

                    search_button.click(
                        fn=search_all_sources,
                        inputs=query_input,
                        outputs=[status_output, results_output],
                        show_progress="full",
                    )
                    query_input.submit(
                        fn=search_all_sources,
                        inputs=query_input,
                        outputs=[status_output, results_output],
                        show_progress="full",
                    )

                with gr.Tab("Summarize"):
                    source_text = gr.Textbox(
                        label="Paper text or abstract",
                        placeholder="Paste an abstract, excerpt, or research notes...",
                        lines=10,
                    )
                    summarize_button = gr.Button("Summarize with AI", variant="primary")
                    summary_output = gr.Textbox(
                        label="Summary",
                        lines=6,
                        interactive=False,
                    )
                    summarize_button.click(
                        fn=summarize_with_modal,
                        inputs=source_text,
                        outputs=summary_output,
                        show_progress="full",
                    )

                with gr.Tab("About"):
                    gr.Markdown(
                        """
                        Scholar Lens is a universal research assistant for discovering papers
                        across Semantic Scholar, arXiv, and PubMed from one focused interface.

                        Search results are normalized into a compact table with title, year,
                        source, authors, and citation information. Rows open the source paper
                        page and are ready for a future detail view.
                        """
                    )

    return app


if __name__ == "__main__":
    build_app().queue().launch()
