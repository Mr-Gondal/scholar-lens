from __future__ import annotations

import html
import os
import re
import textwrap
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import gradio as gr
import requests

APP_TITLE = "Scholar Lens"
SEARCH_LIMIT_PER_SOURCE = 10
REQUEST_TIMEOUT_SECONDS = 15
# Identifies us to the OpenAlex / Crossref "polite pool" for faster, more
# reliable responses. Replace with your own email if you like.
CONTACT_EMAIL = "hussainharis946@gmail.com"
MODAL_SUMMARIZE_URL = os.getenv("MODAL_SUMMARIZE_URL", "").strip()
MODAL_SYNTHESIZE_URL = os.getenv("MODAL_SYNTHESIZE_URL", "").strip()
MODAL_API_TOKEN = os.getenv("SCHOLAR_LENS_MODAL_TOKEN", "").strip()
# How many retrieved papers (that actually have an abstract) to feed the model.
SYNTHESIS_PAPER_COUNT = 6
# Results shown per page in the Search results table.
RESULTS_PER_PAGE = 10
# Trim very long abstracts so the grounded prompt stays a reasonable size.
MAX_ABSTRACT_CHARS = 1400


@dataclass(frozen=True)
class PaperResult:
    title: str
    year: str
    source: str
    authors: str
    citations: str
    url: str
    abstract: str = ""
    doi: str = ""


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


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Rebuild plain text from OpenAlex's abstract inverted index.

    OpenAlex never returns a plain ``abstract`` field; it returns a
    ``{word: [positions]}`` map. We sort words back into reading order.
    """
    if not inverted_index:
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for position in positions:
            positioned.append((position, word))
    positioned.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in positioned)


def _strip_markup(text: str) -> str:
    """Crossref abstracts are JATS XML; reduce them to plain text."""
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").split())


def _normalize_doi(value: str) -> str:
    doi = (value or "").lower().strip()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)


def search_openalex(query: str) -> tuple[list[PaperResult], str | None]:
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": SEARCH_LIMIT_PER_SOURCE,
        "filter": "has_abstract:true",
        "mailto": CONTACT_EMAIL,
    }
    try:
        payload = _request_json(url, params)
    except requests.RequestException:
        return [], "OpenAlex is unavailable right now."

    results: list[PaperResult] = []
    for work in payload.get("results", []):
        authors = [
            authorship.get("author", {}).get("display_name", "")
            for authorship in work.get("authorships", [])
        ]
        results.append(
            PaperResult(
                title=_safe_text(work.get("title"), "Untitled paper"),
                year=_safe_text(work.get("publication_year")),
                source="OpenAlex",
                authors=_shorten_authors(authors),
                citations=_safe_text(work.get("cited_by_count"), "0"),
                url=_safe_text(work.get("doi") or work.get("id"), "#"),
                abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
                doi=_normalize_doi(work.get("doi", "")),
            )
        )
    return results, None


def _crossref_year(item: dict[str, Any]) -> str:
    for key in ("published", "published-print", "published-online", "issued"):
        parts = item.get(key, {}).get("date-parts", [])
        if parts and parts[0] and parts[0][0]:
            return str(parts[0][0])
    return "Unknown"


def search_crossref(query: str) -> tuple[list[PaperResult], str | None]:
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": SEARCH_LIMIT_PER_SOURCE,
        "filter": "has-abstract:true",
        "mailto": CONTACT_EMAIL,
    }
    try:
        payload = _request_json(url, params)
    except requests.RequestException:
        return [], "Crossref is unavailable right now."

    results: list[PaperResult] = []
    for item in payload.get("message", {}).get("items", []):
        title_list = item.get("title") or ["Untitled paper"]
        authors = [
            f"{author.get('given', '')} {author.get('family', '')}".strip()
            for author in item.get("author", [])
        ]
        results.append(
            PaperResult(
                title=_safe_text(title_list[0], "Untitled paper"),
                year=_crossref_year(item),
                source="Crossref",
                authors=_shorten_authors(authors),
                citations=_safe_text(item.get("is-referenced-by-count"), "0"),
                url=_safe_text(item.get("URL"), "#"),
                abstract=_strip_markup(item.get("abstract", "")),
                doi=_normalize_doi(item.get("DOI", "")),
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
        "OpenAlex": ("openalex", "🌐"),
        "Crossref": ("crossref", "🔗"),
        "arXiv": ("arxiv", "📐"),
        "PubMed": ("pubmed", "🧬"),
    }
    class_name, icon = config.get(source, ("default", "📄"))
    return (
        f'<span class="source-badge {class_name}">'
        f'<span class="badge-icon">{icon}</span>{html.escape(source)}'
        f'</span>'
    )


def _render_results_table(results: list[PaperResult], start_index: int = 0) -> str:
    if not results:
        return """
        <div class="empty-state">
            <div class="empty-crest">📚</div>
            <h3>Your library awaits</h3>
            <p>Run a search to populate your scholarly results below.</p>
        </div>
        """
    rows = []
    for offset, result in enumerate(results):
        result_index = start_index + offset
        number = result_index + 1
        safe_url = html.escape(result.url, quote=True)
        rows.append(
            textwrap.dedent(
                f"""
                <tr
                    id="row-{result_index}"
                    class="result-row"
                    role="button"
                    tabindex="0"
                    aria-label="Select paper {number}: {html.escape(result.title, quote=True)}"
                    onclick="selectPaper(event, {result_index})"
                    onkeydown="selectPaperFromKey(event, {result_index})"
                >
                    <td class="num-cell">{number}</td>
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
                    <th style="width: 40px;">#</th>
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


def _page_view(results: list[PaperResult], page: int) -> tuple[str, str, int]:
    """Return (table_html, page_label, clamped_page) for one page of results."""
    total_pages = max(1, (len(results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)
    page = max(0, min(page, total_pages - 1))
    start = page * RESULTS_PER_PAGE
    chunk = results[start:start + RESULTS_PER_PAGE]
    table = _render_results_table(chunk, start_index=start)
    label = f"Page {page + 1} of {total_pages} · {len(results)} papers" if results else ""
    return table, label, page


def _dedupe_results(results: list[PaperResult]) -> list[PaperResult]:
    """Drop duplicate papers that appear in more than one source.

    Papers are matched on a normalized title; the first occurrence wins so we
    keep the source order produced by the search functions.
    """
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[PaperResult] = []
    for result in results:
        doi = _normalize_doi(result.doi)
        if doi and doi in seen_dois:
            continue
        title_key = re.sub(r"[^a-z0-9]+", "", result.title.lower())
        if title_key and title_key in seen_titles:
            continue
        if doi:
            seen_dois.add(doi)
        if title_key:
            seen_titles.add(title_key)
        unique.append(result)
    return unique


def _collect_results(query: str) -> tuple[list[PaperResult], list[str]]:
    """Query every source in parallel, then de-duplicate and sort by year."""
    search_functions = [search_openalex, search_crossref, search_arxiv, search_pubmed]
    results: list[PaperResult] = []
    warnings: list[str] = []

    with ThreadPoolExecutor(max_workers=len(search_functions)) as executor:
        future_to_search = {executor.submit(fn, query): fn.__name__ for fn in search_functions}
        for future in as_completed(future_to_search):
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
    return results, warnings


def _selector_choices(results: list[PaperResult]) -> list[str]:
    return [f"{index + 1}. {result.title}" for index, result in enumerate(results)]


def search_all_sources(query: str):
    clean_query = query.strip()
    if not clean_query:
        table, label, page = _page_view([], 0)
        return (
            "Enter a research topic to search OpenAlex, Crossref, arXiv, and PubMed.",
            table,
            [],
            gr.update(choices=[], value=None),
            label,
            page,
        )

    results, warnings = _collect_results(clean_query)

    if warnings and results:
        status = f"✓ Found **{len(results)}** papers. " + " ".join(warnings)
    elif warnings:
        status = " ".join(warnings)
    else:
        status = f"✓ Found **{len(results)}** papers across all sources."

    if not results:
        status = "No papers found. Try a broader research topic or a different phrase."

    table, label, page = _page_view(results, 0)
    return (
        status,
        table,
        results,
        gr.update(choices=_selector_choices(results), value=None),
        label,
        page,
    )


def change_page(results: list[PaperResult], page: int, delta: int) -> tuple[str, str, int]:
    return _page_view(results, (page or 0) + delta)


def _modal_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {MODAL_API_TOKEN}"}


def _modal_config_error(endpoint_url: str) -> str | None:
    if not endpoint_url:
        return (
            "The AI endpoint is not configured. Set the Modal endpoint URL "
            "environment variable before using this feature."
        )
    if not MODAL_API_TOKEN:
        return (
            "The AI endpoint token is not configured. Set "
            "SCHOLAR_LENS_MODAL_TOKEN before using this feature."
        )
    return None


def summarize_with_modal(text: str) -> str:
    if not text or len(text.strip()) < 50:
        return "Please provide a longer abstract or paper text to summarize."
    config_error = _modal_config_error(MODAL_SUMMARIZE_URL)
    if config_error:
        return config_error
    try:
        response = requests.post(
            MODAL_SUMMARIZE_URL,
            json={"text": text},
            headers=_modal_headers(),
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


def _papers_for_synthesis(results: list[PaperResult]) -> list[PaperResult]:
    """Pick the top results that actually carry an abstract to ground on."""
    with_abstract = [paper for paper in results if paper.abstract.strip()]
    return with_abstract[:SYNTHESIS_PAPER_COUNT]


def _build_synthesis_context(papers: list[PaperResult]) -> str:
    blocks = []
    for index, paper in enumerate(papers, start=1):
        abstract = paper.abstract.strip()
        if len(abstract) > MAX_ABSTRACT_CHARS:
            abstract = abstract[:MAX_ABSTRACT_CHARS].rstrip() + "..."
        blocks.append(
            f"[{index}] Title: {paper.title}\n"
            f"Source: {paper.source} ({paper.year})\n"
            f"Abstract: {abstract}"
        )
    return "\n\n".join(blocks)


def _render_references(papers: list[PaperResult]) -> str:
    if not papers:
        return ""
    items = []
    for index, paper in enumerate(papers, start=1):
        safe_url = html.escape(paper.url, quote=True)
        items.append(
            f'<li class="ref-item">'
            f'<span class="ref-num">[{index}]</span>'
            f'<span class="ref-body">'
            f'<a class="ref-link" href="{safe_url}" target="_blank" rel="noopener noreferrer">{html.escape(paper.title)}</a>'
            f'<span class="ref-meta">{_source_badge(paper.source)} · {html.escape(paper.year)} · {html.escape(paper.authors)}</span>'
            f'</span></li>'
        )
    return (
        '<div class="refs-shell"><h3>📚 Sources the answer is grounded in</h3>'
        f'<ul class="refs-list">{"".join(items)}</ul></div>'
    )


def synthesize_with_modal(question: str, context: str) -> str:
    config_error = _modal_config_error(MODAL_SYNTHESIZE_URL)
    if config_error:
        return config_error
    try:
        response = requests.post(
            MODAL_SYNTHESIZE_URL,
            json={"question": question, "context": context},
            headers=_modal_headers(),
            timeout=180,
        )
        response.raise_for_status()
    except requests.Timeout:
        return (
            "The AI synthesizer timed out (the model may be cold-starting). "
            "Please try again in a few seconds."
        )
    except requests.RequestException:
        return "The AI synthesizer is unavailable right now. Please try again shortly."

    try:
        payload = response.json()
    except ValueError:
        return "The AI synthesizer returned an unexpected response. Please try again shortly."

    if isinstance(payload, dict):
        return _safe_text(payload.get("answer") or payload.get("error"), "No answer returned.")
    return _safe_text(payload, "No answer returned.")


def ask_scholar_lens(question: str) -> tuple[str, str]:
    """Search every source, then have the small model answer with citations."""
    clean_question = question.strip()
    if not clean_question:
        return "Enter a research question to begin.", ""

    results, warnings = _collect_results(clean_question)
    if not results:
        note = " ".join(warnings) if warnings else ""
        return (
            f"No papers were found for that question. Try rephrasing it.\n\n{note}".strip(),
            "",
        )

    papers = _papers_for_synthesis(results)
    if not papers:
        return (
            "Papers were found, but none included an abstract to reason over. "
            "Try a broader or differently worded question.",
            "",
        )

    context = _build_synthesis_context(papers)
    answer = synthesize_with_modal(clean_question, context)
    return answer, _render_references(papers)


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
    selected_index: Any,
    results: list[PaperResult],
) -> tuple[str, str, str, gr.Tabs]:
    try:
        index = int(selected_index)
    except (TypeError, ValueError):
        index = None
    paper_text, load_status, _, tab_update = load_selected_paper(index, results)
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


def summarize_row_selection(
    selected_index: Any,
    results: list[PaperResult],
) -> tuple[str, str, str, gr.Tabs, gr.Dropdown]:
    paper_text, load_status, summary, tab_update = summarize_selected_paper(
        selected_index,
        results,
    )
    try:
        dropdown_value = int(selected_index)
    except (TypeError, ValueError):
        dropdown_value = None
    return (
        paper_text,
        load_status,
        summary,
        tab_update,
        gr.update(value=dropdown_value),
    )


def clear_search():
    table, label, page = _page_view([], 0)
    return (
        "Enter a research topic to begin.",
        table,
        [],
        gr.update(choices=[], value=None),
        label,
        page,
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
.num-cell { color: var(--sl-muted); text-align: center; font-variant-numeric: tabular-nums; }

/* ===== PAGINATION ===== */
.page-row { align-items: center; justify-content: center; gap: 14px; margin-top: 12px; }
.page-label { text-align: center; color: var(--sl-muted) !important; min-width: 180px; }

.source-badge.openalex { color: #22d3ee !important; background: rgba(34, 211, 238, 0.1); border: 1px solid rgba(34, 211, 238, 0.3); }
.source-badge.crossref { color: #fbbf24 !important; background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); }
.source-badge.arxiv { color: #f87171 !important; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); }
.source-badge.pubmed { color: #34d399 !important; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); }

/* ===== ASK / SYNTHESIS ===== */
.ask-intro { color: var(--sl-muted) !important; margin-bottom: 6px; }
.answer-card {
    background: var(--sl-panel) !important;
    border: 1px solid var(--sl-border);
    border-left: 4px solid var(--sl-accent);
    border-radius: 12px;
    padding: 20px 24px !important;
    margin-top: 16px;
    line-height: 1.65;
    font-size: 15px;
}
.answer-card p { color: var(--sl-text) !important; }

.refs-shell {
    margin-top: 18px;
    border: 1px solid var(--sl-border);
    border-radius: 12px;
    background: var(--sl-panel);
    padding: 16px 20px;
}
.refs-shell h3 {
    margin: 0 0 12px;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--sl-accent-bright);
}
.refs-list { list-style: none; margin: 0; padding: 0; }
.ref-item { display: flex; gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--sl-border-soft); }
.ref-item:last-child { border-bottom: none; }
.ref-num { color: var(--sl-accent-bright); font-weight: 700; flex-shrink: 0; }
.ref-body { display: flex; flex-direction: column; gap: 4px; }
.ref-link { color: var(--sl-text-bright) !important; font-weight: 600; text-decoration: none; }
.ref-link:hover { color: var(--sl-accent-bright) !important; text-decoration: underline; }
.ref-meta { color: var(--sl-muted); font-size: 12px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

/* Remove system buttons and footers */
.settings, footer, .show-api { display: none !important; }

.hidden-component { display: none !important; }
"""

# Injected into the page <head> at launch time (Gradio 6 takes `head` on
# launch(), not on Blocks()). Powers click-to-select on the results table.
HEAD_SCRIPT = """
<script>
function selectPaper(event, index) {
    if (event && event.target && event.target.closest('a')) {
        return;
    }
    const input = document.querySelector('#hidden_index_input textarea');
    if (input) {
        input.value = index;
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    document.querySelectorAll('.result-row').forEach(r => r.classList.remove('selected'));
    const row = document.getElementById('row-' + index);
    if (row) row.classList.add('selected');
    const radio = document.getElementById('radio-' + index);
    if (radio) radio.checked = true;
    setTimeout(() => {
        const btn = document.querySelector('#hidden_select_btn button');
        if (btn) btn.click();
    }, 50);
}

function selectPaperFromKey(event, index) {
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectPaper(event, index);
    }
}
</script>
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

    with gr.Blocks(title=APP_TITLE) as app:
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
                                <span class="header-chip">OpenAlex</span>
                                <span class="header-chip">Crossref</span>
                                <span class="header-chip">arXiv</span>
                                <span class="header-chip">PubMed</span>
                            </div>
                        </div>
                    </div>
                </header>
                """
            )

            papers_state = gr.State([])
            page_state = gr.State(0)

            with gr.Tabs(selected="ask") as app_tabs:
                with gr.Tab("💬  Ask", id="ask"):
                    gr.Markdown(
                        "Ask a research question. Scholar Lens searches **OpenAlex, "
                        "Crossref, arXiv, and PubMed**, then a 24B open model "
                        "(Mistral Small 3.1) writes a synthesized, **cited** answer grounded "
                        "only in the retrieved abstracts.",
                        elem_classes=["ask-intro"],
                    )
                    with gr.Row():
                        question_input = gr.Textbox(
                            label="",
                            placeholder="e.g. What are the main approaches to early Alzheimer's detection from MRI, and where do they disagree?",
                            scale=5,
                            container=False,
                            lines=2,
                        )
                        ask_button = gr.Button("💬 Ask", variant="primary", scale=1)

                    answer_output = gr.Markdown(
                        "Your grounded, cited answer will appear here.",
                        elem_classes=["answer-card"],
                    )
                    references_output = gr.HTML()

                    ask_button.click(
                        fn=ask_scholar_lens,
                        inputs=question_input,
                        outputs=[answer_output, references_output],
                        show_progress="full",
                    )
                    question_input.submit(
                        fn=ask_scholar_lens,
                        inputs=question_input,
                        outputs=[answer_output, references_output],
                        show_progress="full",
                    )

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

                    with gr.Row(elem_classes=["page-row"]):
                        prev_button = gr.Button("← Prev", scale=0, min_width=100)
                        page_label = gr.Markdown("", elem_classes=["page-label"])
                        next_button = gr.Button("Next →", scale=0, min_width=100)

                    with gr.Row(elem_classes=["action-row"]):
                        paper_selector = gr.Dropdown(
                            label="Select a paper to summarize",
                            choices=[],
                            type="index",
                            interactive=True,
                            scale=4,
                        )
                        with gr.Column(scale=1):
                            summarize_selected_button = gr.Button("✨ Summarize Now", variant="primary")

                    search_outputs = [
                        status_output,
                        results_output,
                        papers_state,
                        paper_selector,
                        page_label,
                        page_state,
                    ]
                    search_button.click(
                        fn=search_all_sources,
                        inputs=query_input,
                        outputs=search_outputs,
                        show_progress="full",
                    )
                    query_input.submit(
                        fn=search_all_sources,
                        inputs=query_input,
                        outputs=search_outputs,
                        show_progress="full",
                    )

                    prev_button.click(
                        fn=lambda results, page: change_page(results, page, -1),
                        inputs=[papers_state, page_state],
                        outputs=[results_output, page_label, page_state],
                    )
                    next_button.click(
                        fn=lambda results, page: change_page(results, page, 1),
                        inputs=[papers_state, page_state],
                        outputs=[results_output, page_label, page_state],
                    )

                    hidden_selected_index = gr.Textbox(
                        label="Selected row index",
                        elem_id="hidden_index_input",
                        elem_classes=["hidden-component"],
                    )
                    hidden_select_button = gr.Button(
                        "Select row",
                        elem_id="hidden_select_btn",
                        elem_classes=["hidden-component"],
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
                    
                    # Native dropdown selection (type="index" gives the chosen
                    # paper's index directly) — no JS, no race conditions.
                    summarize_selected_button.click(
                        fn=summarize_selected_paper,
                        inputs=[paper_selector, papers_state],
                        outputs=[source_text, load_status_output, summary_output, app_tabs],
                        show_progress="full",
                    )
                    hidden_select_button.click(
                        fn=summarize_row_selection,
                        inputs=[hidden_selected_index, papers_state],
                        outputs=[
                            source_text,
                            load_status_output,
                            summary_output,
                            app_tabs,
                            paper_selector,
                        ],
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
                                It performs real-time searches across <strong>OpenAlex</strong>,
                                <strong>Crossref</strong>, <strong>arXiv</strong>, and <strong>PubMed</strong>
                                using parallel processing, then uses a 24B open language model
                                (<strong>Mistral Small 3.1</strong>, hosted on Modal) to do the heavy lifting.
                            </p>
                            <p>
                                In the <strong>Ask</strong> tab the model answers your research
                                question with a synthesized, <strong>cited</strong> response grounded
                                only in the retrieved abstracts &mdash; so it compares findings across
                                papers without inventing sources. The <strong>Summarize</strong> tab
                                condenses any single paper or pasted text.
                            </p>
                            <p style="margin-top: 16px; font-size: 13px; color: var(--sl-muted);">
                                Built for the Hugging Face <em>Build Small</em> hackathon &middot;
                                model &le; 32B parameters.
                            </p>
                        </div>
                        """
                    )
            
            clear_button.click(
                fn=clear_search,
                outputs=[
                    status_output,
                    results_output,
                    papers_state,
                    paper_selector,
                    page_label,
                    page_state,
                    source_text,
                    summary_output,
                ],
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
        head=HEAD_SCRIPT,
    )
