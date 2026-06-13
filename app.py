from __future__ import annotations

import html
import os
import re
import csv
import textwrap
import time
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import gradio as gr
import requests

APP_TITLE = "Scholar Lens"
SEARCH_LIMIT_PER_SOURCE = 10
REQUEST_TIMEOUT_SECONDS = 15
REQUEST_RETRY_ATTEMPTS = 3
REQUEST_RETRY_BACKOFF_SECONDS = 0.6
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
# Identifies us to the OpenAlex / Crossref "polite pool" for faster, more
# reliable responses. Replace with your own email if you like.
CONTACT_EMAIL = os.getenv("SCHOLAR_LENS_CONTACT_EMAIL", "hussainharis946@gmail.com").strip()
MODAL_SUMMARIZE_URL = os.getenv("MODAL_SUMMARIZE_URL", "").strip()
MODAL_SYNTHESIZE_URL = os.getenv("MODAL_SYNTHESIZE_URL", "").strip()
MODAL_API_TOKEN = os.getenv("SCHOLAR_LENS_MODAL_TOKEN", "").strip()
# How many retrieved papers (that actually have an abstract) to feed the model.
SYNTHESIS_PAPER_COUNT = 6
# Results shown per page in the Search results table.
RESULTS_PER_PAGE = 10
# Trim very long abstracts so the grounded prompt stays a reasonable size.
MAX_ABSTRACT_CHARS = 1400
MAX_ABSTRACT_TOKENS = 350
SEARCH_QUERY_CHAR_LIMIT = 300
ASK_QUESTION_CHAR_LIMIT = 1200
ASK_QUESTION_TOKEN_LIMIT = 300
SUMMARY_INPUT_CHAR_LIMIT = 60000
SUMMARY_INPUT_TOKEN_LIMIT = 15000
SYNTHESIS_CONTEXT_CHAR_LIMIT = 28000
SYNTHESIS_CONTEXT_TOKEN_LIMIT = 7000
DEFAULT_ASK_ANSWER = "Your grounded, cited answer will appear here."
DEFAULT_LOAD_STATUS = "Select a paper to summarize."
DEFAULT_PAPER_CHAT_ANSWER = "Ask a question about the selected or pasted paper."
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
FUZZY_TITLE_THRESHOLD = 0.94
EXPORT_DIR = Path(tempfile.gettempdir()) / "scholar-lens-exports"
SAMPLE_QUESTIONS = [
    "Where do papers disagree on aerosol-cloud interaction uncertainty?",
    "How is machine learning improving satellite precipitation estimates?",
    "What are recent methods for detecting atmospheric rivers?",
    "How do climate models represent urban heat island effects?",
]
SEARCH_STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "as",
    "be",
    "being",
    "between",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "main",
    "of",
    "on",
    "or",
    "recent",
    "show",
    "the",
    "their",
    "these",
    "to",
    "used",
    "using",
    "what",
    "where",
    "which",
    "with",
}


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


def _rough_token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text or ""))


def _trim_to_budget(text: str, max_chars: int, max_tokens: int) -> str:
    trimmed = (text or "").strip()
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rstrip()
    while _rough_token_count(trimmed) > max_tokens and trimmed:
        next_length = max(1, int(len(trimmed) * 0.85))
        trimmed = trimmed[:next_length].rsplit(" ", 1)[0].rstrip()
    return trimmed


def _text_limit_error(
    text: str,
    label: str,
    max_chars: int,
    max_tokens: int,
) -> str | None:
    if len(text) > max_chars:
        return (
            f"{label} is too long for this demo. Please keep it under "
            f"{max_chars:,} characters."
        )
    token_count = _rough_token_count(text)
    if token_count > max_tokens:
        return (
            f"{label} is too long for this demo. Please keep it under roughly "
            f"{max_tokens:,} tokens; this input is about {token_count:,} tokens."
        )
    return None


def _search_terms(query: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", query.lower())
    terms: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in SEARCH_STOPWORDS or word in seen:
            continue
        seen.add(word)
        terms.append(word)
    return terms


def _extract_search_query(question: str) -> str:
    terms = _search_terms(question)
    if not terms:
        return question.strip()
    return " ".join(terms[:8])


def _year_value(item: PaperResult) -> int:
    return int(item.year) if item.year.isdigit() else 0


def _citation_value(item: PaperResult) -> int:
    return int(item.citations) if item.citations.isdigit() else 0


def _relevance_score(item: PaperResult, query: str) -> int:
    terms = _search_terms(query)
    if not terms:
        return 0
    title = item.title.lower()
    abstract = item.abstract.lower()
    authors = item.authors.lower()
    score = 0
    for term in terms:
        if term in title:
            score += 8
        if term in abstract:
            score += 3
        if term in authors:
            score += 1
    return score


def _rank_results(results: list[PaperResult], query: str) -> list[PaperResult]:
    return sorted(
        results,
        key=lambda item: (
            _relevance_score(item, query),
            _year_value(item),
            min(_citation_value(item), 5000),
        ),
        reverse=True,
    )


def _arxiv_search_query(query: str) -> str:
    terms = _search_terms(query)[:6]
    if not terms:
        return f"all:{query}"
    return " OR ".join(f"ti:{term} OR abs:{term}" for term in terms)


def _pubmed_search_query(query: str) -> str:
    terms = _search_terms(query)[:8]
    if not terms:
        return query
    return " OR ".join(f'{term}[Title/Abstract]' for term in terms)


def _result_quality(result: PaperResult) -> tuple[int, int, int]:
    return (
        1 if result.abstract.strip() else 0,
        _citation_value(result),
        _year_value(result),
    )


def _titles_match(left: str, right: str) -> bool:
    left_key = re.sub(r"[^a-z0-9]+", "", left.lower())
    right_key = re.sub(r"[^a-z0-9]+", "", right.lower())
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    return SequenceMatcher(None, left_key, right_key).ratio() >= FUZZY_TITLE_THRESHOLD


def _render_result_insights(results: list[PaperResult]) -> str:
    if not results:
        return ""

    source_counts = Counter(result.source for result in results)
    years = [_year_value(result) for result in results if _year_value(result)]
    year_range = f"{min(years)}-{max(years)}" if years else "Unknown"
    abstracts = sum(1 for result in results if result.abstract.strip())
    top_source, top_source_count = source_counts.most_common(1)[0]
    cards = [
        ("Papers", f"{len(results)}", "Deduplicated results"),
        ("Sources", f"{len(source_counts)}", ", ".join(sorted(source_counts))),
        ("Year Range", year_range, "Publication years"),
        ("Abstracts", f"{abstracts}", "Ready for AI grounding"),
        ("Top Source", top_source, f"{top_source_count} papers"),
    ]
    rendered_cards = "".join(
        f'<div class="insight-card"><span>{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong><small>{html.escape(detail)}</small></div>"
        for label, value, detail in cards
    )
    return f'<div class="insight-grid">{rendered_cards}</div>'


def _abstract_snippet(text: str, max_chars: int = 220) -> str:
    snippet = " ".join((text or "").split())
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."
    return snippet


def _ensure_export_dir() -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return EXPORT_DIR


def export_results_csv(results: list[PaperResult]) -> str | None:
    if not results:
        return None
    path = _ensure_export_dir() / "scholar_lens_results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["title", "year", "source", "authors", "citations", "url", "doi", "abstract"])
        for result in results:
            writer.writerow(
                [
                    result.title,
                    result.year,
                    result.source,
                    result.authors,
                    result.citations,
                    result.url,
                    result.doi,
                    result.abstract,
                ]
            )
    return str(path)


def export_summary_markdown(source_text: str, results_text: str, summary: str) -> str | None:
    if not source_text.strip() and not results_text.strip() and not summary.strip():
        return None
    path = _ensure_export_dir() / "scholar_lens_summary.md"
    path.write_text(
        "# Scholar Lens Summary\n\n"
        "## Source Context\n\n"
        f"{source_text.strip() or '_No source context provided._'}\n\n"
        "## Results / Findings\n\n"
        f"{results_text.strip() or '_No results/findings section provided._'}\n\n"
        "## AI Summary\n\n"
        f"{summary.strip() or '_No summary generated yet._'}\n",
        encoding="utf-8",
    )
    return str(path)


def _combine_paper_context(source_text: str, results_text: str = "") -> str:
    parts = []
    if source_text and source_text.strip():
        parts.append(source_text.strip())
    if results_text and results_text.strip():
        parts.append(f"Results / Findings:\n{results_text.strip()}")
    return "\n\n".join(parts)


def summarize_paper_context(source_text: str, results_text: str) -> str:
    return summarize_with_modal(_combine_paper_context(source_text, results_text))


def chat_about_paper(source_text: str, results_text: str, question: str) -> str:
    clean_question = (question or "").strip()
    context = _combine_paper_context(source_text, results_text)
    if not context.strip():
        return "Load a paper or paste paper text before asking a question."
    if not clean_question:
        return "Ask a specific question about the paper."
    limit_error = _text_limit_error(
        clean_question,
        "Paper question",
        ASK_QUESTION_CHAR_LIMIT,
        ASK_QUESTION_TOKEN_LIMIT,
    )
    if limit_error:
        return limit_error
    context = _trim_to_budget(
        context,
        SYNTHESIS_CONTEXT_CHAR_LIMIT,
        SYNTHESIS_CONTEXT_TOKEN_LIMIT,
    )
    prompt_question = (
        f"{clean_question}\n\n"
        "Answer using only this paper context. If the answer depends on the "
        "results/findings section and it is not present, say that clearly."
    )
    return synthesize_with_modal(prompt_question, f"[1] Paper context:\n{context}")


def ask_example(question: str) -> tuple[str, str, str]:
    answer, references = ask_scholar_lens(question)
    return question, answer, references


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


def _get_with_retries(
    url: str,
    params: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> requests.Response:
    request_headers = {
        "User-Agent": "ScholarLens/1.0 (academic search app)",
        **(headers or {}),
    }
    last_error: requests.RequestException | None = None
    for attempt in range(REQUEST_RETRY_ATTEMPTS):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers=request_headers,
            )
            if response.status_code not in RETRY_STATUS_CODES:
                response.raise_for_status()
                return response
            last_error = requests.HTTPError(
                f"{response.status_code} response from {url}",
                response=response,
            )
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last_error = exc
            response = getattr(exc, "response", None)
            if response is not None and response.status_code not in RETRY_STATUS_CODES:
                raise
        if attempt < REQUEST_RETRY_ATTEMPTS - 1:
            time.sleep(REQUEST_RETRY_BACKOFF_SECONDS * (attempt + 1))
    if last_error:
        raise last_error
    raise requests.RequestException(f"Request failed for {url}")


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = _get_with_retries(url, params, headers={"Accept": "application/json"})
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
        "search_query": _arxiv_search_query(query),
        "start": 0,
        "max_results": SEARCH_LIMIT_PER_SOURCE,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        response = _get_with_retries(url, params)
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
        "term": _pubmed_search_query(query),
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
    response = _get_with_retries(
        fetch_url,
        params={
            "db": "pubmed",
            "id": ",".join(paper_ids),
            "retmode": "xml",
        },
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
                    <td class="action-cell"><span class="row-action">Summarize</span><a class="paper-link" href="{safe_url}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()">Open ↗</a></td>
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
                    <th>Action</th>
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


def _pagination_updates(results: list[PaperResult], page: int) -> tuple[gr.Button, gr.Button]:
    total_pages = max(1, (len(results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)
    has_results = bool(results)
    return (
        gr.update(interactive=has_results and page > 0),
        gr.update(interactive=has_results and page < total_pages - 1),
    )


def _dedupe_results(results: list[PaperResult]) -> list[PaperResult]:
    """Drop duplicate papers that appear in more than one source.

    DOI wins first, then fuzzy title matching. When duplicates are found, keep
    the version with the strongest metadata rather than whichever API returned
    first.
    """
    unique: list[PaperResult] = []
    for result in results:
        doi = _normalize_doi(result.doi)
        duplicate_index = None
        for index, existing in enumerate(unique):
            existing_doi = _normalize_doi(existing.doi)
            if doi and existing_doi and doi == existing_doi:
                duplicate_index = index
                break
            if _titles_match(result.title, existing.title):
                duplicate_index = index
                break

        if duplicate_index is None:
            unique.append(result)
            continue

        existing = unique[duplicate_index]
        if _result_quality(result) > _result_quality(existing):
            unique[duplicate_index] = result
    return unique


def _collect_results(query: str) -> tuple[list[PaperResult], list[str]]:
    """Query every source in parallel, then de-duplicate and rank results."""
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

    return _rank_results(results, query), warnings


def _selector_choices(results: list[PaperResult]) -> list[str]:
    return [f"{index + 1}. {result.title}" for index, result in enumerate(results)]


def search_all_sources(query: str):
    clean_query = query.strip()
    if not clean_query:
        table, label, page = _page_view([], 0)
        prev_update, next_update = _pagination_updates([], page)
        return (
            "Enter a research topic to search OpenAlex, Crossref, arXiv, and PubMed.",
            table,
            "",
            [],
            gr.update(choices=[], value=None),
            label,
            page,
            prev_update,
            next_update,
            None,
        )
    if len(clean_query) > SEARCH_QUERY_CHAR_LIMIT:
        table, label, page = _page_view([], 0)
        prev_update, next_update = _pagination_updates([], page)
        return (
            f"Search topic is too long. Please keep it under {SEARCH_QUERY_CHAR_LIMIT} characters.",
            table,
            "",
            [],
            gr.update(choices=[], value=None),
            label,
            page,
            prev_update,
            next_update,
            None,
        )

    results, warnings = _collect_results(clean_query)

    if warnings and results:
        status = f"✓ Found **{len(results)}** papers. " + " ".join(warnings)
    elif warnings:
        status = " ".join(warnings)
    else:
        status = f"✓ Found **{len(results)}** papers across all sources."

    if not results:
        if warnings:
            status = "No papers could be loaded because the source APIs are unavailable right now. " + " ".join(warnings)
        else:
            status = "No papers found. Try a broader research topic or a different phrase."

    table, label, page = _page_view(results, 0)
    prev_update, next_update = _pagination_updates(results, page)
    results_csv = export_results_csv(results) if results else None
    return (
        status,
        table,
        _render_result_insights(results),
        results,
        gr.update(choices=_selector_choices(results), value=None),
        label,
        page,
        prev_update,
        next_update,
        results_csv,
    )


def change_page(results: list[PaperResult], page: int, delta: int):
    table, label, clamped_page = _page_view(results, (page or 0) + delta)
    prev_update, next_update = _pagination_updates(results, clamped_page)
    return table, label, clamped_page, prev_update, next_update


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


def _modal_request_error_message(exc: requests.RequestException, label: str) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        detail = payload.get("detail") or payload.get("error")
        if detail:
            return f"{label}: {detail}"
    return f"{label} is unavailable right now. Please try again shortly."


def summarize_with_modal(text: str) -> str:
    if not text or len(text.strip()) < 50:
        return "Please provide a longer abstract or paper text to summarize."
    clean_text = text.strip()
    limit_error = _text_limit_error(
        clean_text,
        "Paper text",
        SUMMARY_INPUT_CHAR_LIMIT,
        SUMMARY_INPUT_TOKEN_LIMIT,
    )
    if limit_error:
        return limit_error
    config_error = _modal_config_error(MODAL_SUMMARIZE_URL)
    if config_error:
        return config_error
    try:
        response = requests.post(
            MODAL_SUMMARIZE_URL,
            json={"text": clean_text},
            headers=_modal_headers(),
            timeout=120,
        )
        response.raise_for_status()
    except requests.Timeout:
        return (
            "The AI summarizer timed out (the model may be cold-starting). "
            "Please try again in a few seconds."
        )
    except requests.RequestException as exc:
        return _modal_request_error_message(exc, "The AI summarizer")

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
        abstract = _trim_to_budget(
            paper.abstract,
            MAX_ABSTRACT_CHARS,
            MAX_ABSTRACT_TOKENS,
        )
        blocks.append(
            f"[{index}] Title: {paper.title}\n"
            f"Source: {paper.source} ({paper.year})\n"
            f"Abstract: {abstract}"
        )
        context = "\n\n".join(blocks)
        if (
            len(context) > SYNTHESIS_CONTEXT_CHAR_LIMIT
            or _rough_token_count(context) > SYNTHESIS_CONTEXT_TOKEN_LIMIT
        ):
            blocks.pop()
            break
    return "\n\n".join(blocks)


def _render_references(papers: list[PaperResult]) -> str:
    if not papers:
        return ""
    items = []
    for index, paper in enumerate(papers, start=1):
        safe_url = html.escape(paper.url, quote=True)
        snippet = _abstract_snippet(paper.abstract)
        items.append(
            f'<li class="ref-item">'
            f'<span class="ref-num">[{index}]</span>'
            f'<span class="ref-body">'
            f'<a class="ref-link" href="{safe_url}" target="_blank" rel="noopener noreferrer">{html.escape(paper.title)}</a>'
            f'<span class="ref-meta">{_source_badge(paper.source)} · {html.escape(paper.year)} · {html.escape(paper.authors)}</span>'
            f'<span class="ref-snippet">{html.escape(snippet)}</span>'
            f'</span></li>'
        )
    return (
        '<div class="refs-shell"><h3>📚 Sources the answer is grounded in</h3>'
        f'<ul class="refs-list">{"".join(items)}</ul></div>'
    )


def synthesize_with_modal(question: str, context: str) -> str:
    question_limit_error = _text_limit_error(
        question,
        "Research question",
        ASK_QUESTION_CHAR_LIMIT,
        ASK_QUESTION_TOKEN_LIMIT,
    )
    if question_limit_error:
        return question_limit_error
    context_limit_error = _text_limit_error(
        context,
        "Synthesis context",
        SYNTHESIS_CONTEXT_CHAR_LIMIT,
        SYNTHESIS_CONTEXT_TOKEN_LIMIT,
    )
    if context_limit_error:
        return context_limit_error
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
    except requests.RequestException as exc:
        return _modal_request_error_message(exc, "The AI synthesizer")

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
    question_limit_error = _text_limit_error(
        clean_question,
        "Research question",
        ASK_QUESTION_CHAR_LIMIT,
        ASK_QUESTION_TOKEN_LIMIT,
    )
    if question_limit_error:
        return question_limit_error, ""

    search_query = _extract_search_query(clean_question)
    results, warnings = _collect_results(search_query)
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
    if not context:
        return (
            "The retrieved abstracts were too large to fit the demo context "
            "budget. Try a narrower or more specific question.",
            "",
        )
    answer = synthesize_with_modal(clean_question, context)
    return f"**Search terms used:** `{search_query}`\n\n{answer}", _render_references(papers)


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
) -> tuple[str, str, str, gr.Tabs, gr.Dropdown, str, str, str]:
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
        "",
        "",
        DEFAULT_PAPER_CHAT_ANSWER,
    )


def summarize_selected_paper_reset_chat(
    selected_index: Any,
    results: list[PaperResult],
) -> tuple[str, str, str, gr.Tabs, str, str, str]:
    paper_text, load_status, summary, tab_update = summarize_selected_paper(
        selected_index,
        results,
    )
    return paper_text, load_status, summary, tab_update, "", "", DEFAULT_PAPER_CHAT_ANSWER


def clear_search():
    table, label, page = _page_view([], 0)
    prev_update, next_update = _pagination_updates([], page)
    return (
        "Enter a research topic to begin.",
        table,
        "",
        [],
        gr.update(choices=[], value=None),
        label,
        page,
        prev_update,
        next_update,
        None,
        "",
        "",
        "",
        "",
        DEFAULT_ASK_ANSWER,
        "",
        DEFAULT_LOAD_STATUS,
        "",
        None,
        "",
        "",
        DEFAULT_PAPER_CHAT_ANSWER,
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
    overflow-x: auto;
    overflow-y: hidden;
}
.results-table { width: 100%; min-width: 860px; border-collapse: collapse; }
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

.status-line {
    color: var(--sl-muted) !important;
    padding: 4px 2px;
}
.insight-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin: 12px 0 14px;
}
.insight-card {
    border: 1px solid var(--sl-border);
    border-radius: 8px;
    background: var(--sl-panel);
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.insight-card span {
    color: var(--sl-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    font-weight: 700;
}
.insight-card strong { color: var(--sl-text-bright); font-size: 20px; line-height: 1.1; }
.insight-card small { color: var(--sl-muted); font-size: 12px; }
.action-row { align-items: end; gap: 12px; }
.gradio-container [role="option"] {
    color: rgb(10, 20, 252) !important;
}
.gradio-container [role="option"]:hover,
.gradio-container [role="option"][aria-selected="true"] {
    color: rgb(10, 20, 252) !important;
    font-weight: 700 !important;
}
.action-cell {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}
.row-action {
    color: var(--sl-accent-bright);
    font-size: 12px;
    font-weight: 700;
}
.btn-crimson button {
    border-color: rgba(239, 68, 68, 0.4) !important;
    color: #fecaca !important;
}
.sample-row {
    gap: 8px;
    margin: 6px 0 12px;
}
.sample-row button {
    white-space: normal !important;
    line-height: 1.25 !important;
}
.summarize-panel {
    border: 1px solid var(--sl-border);
    border-radius: 8px;
    background: var(--sl-panel);
    padding: 16px;
}
.about-card {
    max-width: 820px;
    border: 1px solid var(--sl-border);
    border-radius: 8px;
    background: var(--sl-panel);
    padding: 22px 24px;
    line-height: 1.65;
}
.about-card h2 { margin-top: 0; color: var(--sl-text-bright); }
.download-action { margin-top: 8px; }

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
.ref-snippet { color: var(--sl-text); font-size: 13px; line-height: 1.45; }

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
                            <p>Small-model literature review for atmospheric science</p>
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
                        "Crossref, arXiv, and PubMed**, then Qwen2.5 3B writes "
                        "a synthesized, **cited** answer grounded "
                        "only in the retrieved abstracts.",
                        elem_classes=["ask-intro"],
                    )
                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Research question",
                            placeholder="e.g. What are the main approaches to early Alzheimer's detection from MRI, and where do they disagree?",
                            scale=5,
                            container=False,
                            lines=2,
                            max_length=ASK_QUESTION_CHAR_LIMIT,
                        )
                        ask_button = gr.Button("💬 Ask", variant="primary", scale=1)

                    answer_output = gr.Markdown(
                        DEFAULT_ASK_ANSWER,
                        elem_classes=["answer-card"],
                    )
                    references_output = gr.HTML()

                    with gr.Row(elem_classes=["sample-row"]):
                        for sample_question in SAMPLE_QUESTIONS:
                            sample_button = gr.Button(sample_question, size="sm", scale=1)
                            sample_button.click(
                                fn=lambda sample=sample_question: ask_example(sample),
                                outputs=[question_input, answer_output, references_output],
                                show_progress="full",
                            )

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
                            label="Search topic",
                            placeholder="Enter research topic (e.g., 'Quantum computing in drug discovery')",
                            scale=5,
                            container=False,
                            max_length=SEARCH_QUERY_CHAR_LIMIT,
                        )
                        search_button = gr.Button("🔍 Search", variant="primary", scale=1)
                        with gr.Column(scale=0, min_width=100, elem_classes=["btn-crimson"]):
                            clear_button = gr.Button("Clear")

                    status_output = gr.Markdown("Ready for search.", elem_classes=["status-line"])
                    insights_output = gr.HTML()
                    results_output = gr.HTML(_render_results_table([]))
                    results_download = gr.DownloadButton(
                        "Download Results CSV",
                        value=None,
                        size="sm",
                        elem_classes=["download-action"],
                    )

                    with gr.Row(elem_classes=["page-row"]):
                        prev_button = gr.Button("← Prev", scale=0, min_width=100, interactive=False)
                        page_label = gr.Markdown("", elem_classes=["page-label"])
                        next_button = gr.Button("Next →", scale=0, min_width=100, interactive=False)

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
                        insights_output,
                        papers_state,
                        paper_selector,
                        page_label,
                        page_state,
                        prev_button,
                        next_button,
                        results_download,
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
                        outputs=[results_output, page_label, page_state, prev_button, next_button],
                    )
                    next_button.click(
                        fn=lambda results, page: change_page(results, page, 1),
                        inputs=[papers_state, page_state],
                        outputs=[results_output, page_label, page_state, prev_button, next_button],
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
                        load_status_output = gr.Markdown(DEFAULT_LOAD_STATUS)
                        source_text = gr.Textbox(
                            label="Paper Context / Abstract",
                            lines=10,
                            max_length=SUMMARY_INPUT_CHAR_LIMIT,
                            buttons=["copy"],
                        )
                        results_text = gr.Textbox(
                            label="Results / Findings Section (optional)",
                            placeholder="Paste the paper's results, findings, discussion, or conclusion section here when available.",
                            lines=6,
                            max_length=SUMMARY_INPUT_CHAR_LIMIT,
                            buttons=["copy"],
                        )
                        summarize_button = gr.Button("✨ Summarize with AI", variant="primary")
                        summary_output = gr.Textbox(
                            label="AI Synthesis",
                            lines=8,
                            interactive=False,
                            buttons=["copy"],
                        )
                        summary_download = gr.DownloadButton(
                            "Download Summary Markdown",
                            value=None,
                            size="sm",
                            elem_classes=["download-action"],
                        )
                        chat_question = gr.Textbox(
                            label="Talk with AI about this paper",
                            placeholder="Ask about methods, assumptions, findings, limitations, or what the results mean.",
                            lines=2,
                            max_length=ASK_QUESTION_CHAR_LIMIT,
                        )
                        chat_button = gr.Button("Ask About Paper", variant="primary")
                        chat_output = gr.Markdown(
                            DEFAULT_PAPER_CHAT_ANSWER,
                            elem_classes=["answer-card"],
                        )

                    summarize_button.click(
                        fn=summarize_paper_context,
                        inputs=[source_text, results_text],
                        outputs=summary_output,
                        show_progress="full",
                    )
                    summary_download.click(
                        fn=export_summary_markdown,
                        inputs=[source_text, results_text, summary_output],
                        outputs=summary_download,
                    )
                    chat_button.click(
                        fn=chat_about_paper,
                        inputs=[source_text, results_text, chat_question],
                        outputs=chat_output,
                        show_progress="full",
                    )
                    chat_question.submit(
                        fn=chat_about_paper,
                        inputs=[source_text, results_text, chat_question],
                        outputs=chat_output,
                        show_progress="full",
                    )
                    
                    # Native dropdown selection (type="index" gives the chosen
                    # paper's index directly) — no JS, no race conditions.
                    summarize_selected_button.click(
                        fn=summarize_selected_paper_reset_chat,
                        inputs=[paper_selector, papers_state],
                        outputs=[
                            source_text,
                            load_status_output,
                            summary_output,
                            app_tabs,
                            results_text,
                            chat_question,
                            chat_output,
                        ],
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
                            results_text,
                            chat_question,
                            chat_output,
                        ],
                        show_progress="full",
                    )

                with gr.Tab("📖  About"):
                    gr.HTML(
                        """
                        <div class="about-card">
                            <h2>🎓 About Scholar Lens</h2>
                            <p>
                                <strong>Scholar Lens</strong> is a small-model academic discovery engine
                                built for atmospheric-science literature reviews and research synthesis.
                            </p>
                            <p>
                                It performs real-time searches across <strong>OpenAlex</strong>,
                                <strong>Crossref</strong>, <strong>arXiv</strong>, and <strong>PubMed</strong>
                                using parallel processing, then uses <strong>Qwen2.5 3B</strong>,
                                hosted on Modal, to do the heavy lifting.
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
                                model small enough for the consumer-GPU story.
                            </p>
                        </div>
                        """
                    )
            
            clear_button.click(
                fn=clear_search,
                outputs=[
                    status_output,
                    results_output,
                    insights_output,
                    papers_state,
                    paper_selector,
                    page_label,
                    page_state,
                    prev_button,
                    next_button,
                    results_download,
                    source_text,
                    results_text,
                    summary_output,
                    query_input,
                    question_input,
                    answer_output,
                    references_output,
                    load_status_output,
                    hidden_selected_index,
                    summary_download,
                    chat_question,
                    chat_output,
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
