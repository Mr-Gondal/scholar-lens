from __future__ import annotations

import html
import os
import re
import csv
import itertools
import json
import textwrap
import time
import tempfile
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import gradio as gr
import requests

# Load a local .env (if present) so env vars like SEMANTIC_SCHOLAR_API_KEY work
# for local dev. On Hugging Face Spaces there is no .env; secrets are injected
# as real env vars, so this is a harmless no-op there.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

APP_TITLE = "Scholar Lens"
MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
MODEL_DISPLAY_NAME = "NVIDIA Llama-Nemotron-Nano 8B"
MODEL_PROVIDER_BADGE = "Powered by NVIDIA Nemotron on Modal"
SEARCH_LIMIT_PER_SOURCE = 10
CONSTELLATION_OPENALEX_LIMIT = 200
CONSTELLATION_MAX_EDGES = 700
REQUEST_TIMEOUT_SECONDS = 15
REQUEST_RETRY_ATTEMPTS = 3
REQUEST_RETRY_BACKOFF_SECONDS = 0.6
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
# Identifies us to the OpenAlex / Crossref "polite pool" for faster, more
# reliable responses. Replace with your own email if you like.
CONTACT_EMAIL = os.getenv("SCHOLAR_LENS_CONTACT_EMAIL", "scholar-lens@example.com").strip()
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()
MODAL_SUMMARIZE_URL = os.getenv("MODAL_SUMMARIZE_URL", "").strip()
MODAL_SYNTHESIZE_URL = os.getenv("MODAL_SYNTHESIZE_URL", "").strip()
MODAL_API_TOKEN = os.getenv("SCHOLAR_LENS_MODAL_TOKEN", "").strip()
# Optional Semantic Scholar API key (set via HF Space secret). With a key the
# Graph API is far less rate-limited; without one the source is simply skipped.
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
# How many retrieved papers (that actually have an abstract) to feed the model.
SYNTHESIS_PAPER_COUNT = 6
# Results shown per page in the Search results table.
RESULTS_PER_PAGE = 10
# Trim very long abstracts so the grounded prompt stays a reasonable size.
MAX_ABSTRACT_CHARS = 1400
MAX_ABSTRACT_TOKENS = 350
SEARCH_QUERY_CHAR_LIMIT = 300
MAX_SEARCH_KEYWORDS = 6
ASK_QUESTION_CHAR_LIMIT = 1200
ASK_QUESTION_TOKEN_LIMIT = 300
SUMMARY_INPUT_CHAR_LIMIT = 60000
SUMMARY_INPUT_TOKEN_LIMIT = 15000
SYNTHESIS_CONTEXT_CHAR_LIMIT = 28000
SYNTHESIS_CONTEXT_TOKEN_LIMIT = 7000
DEFAULT_ASK_ANSWER = "Your grounded, cited answer will appear here."
DEFAULT_LOAD_STATUS = "Select a paper to summarize."
DEFAULT_PAPER_CHAT_ANSWER = "Ask a question about the selected or pasted paper."
DEFAULT_COMPARE_ANSWER = "Search papers, choose any two, then compare them with NVIDIA Nemotron."
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
FUZZY_TITLE_THRESHOLD = 0.94
EXPORT_DIR = Path(tempfile.gettempdir()) / "scholar-lens-exports"
SAMPLE_QUESTIONS = [
    "Where do papers disagree on aerosol-cloud interaction uncertainty?",
    "How is machine learning improving satellite precipitation estimates?",
    "What are recent methods for detecting atmospheric rivers?",
    "How do climate models represent urban heat island effects?",
]
RUBRIC_PROOF_POINTS = [
    (
        "Real professor workflow",
        "Built around an atmospheric-science literature-review pain point, not a generic paper search demo.",
    ),
    (
        "Adoption proof",
        "Validated with a real atmospheric-science researcher using their own research questions.",
    ),
    (
        "NVIDIA Nemotron fit",
        f"{MODEL_DISPLAY_NAME} only synthesizes retrieved abstracts, so the model is load-bearing without needing broad world knowledge.",
    ),
    (
        "Product polish",
        "Ask is the default path, with cited answers, source links, pagination, and friendly empty/error states.",
    ),
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
GRAPH_STOPWORDS = SEARCH_STOPWORDS | {
    "analysis",
    "based",
    "brain",
    "connectivity",
    "data",
    "different",
    "functional",
    "human",
    "model",
    "models",
    "network",
    "networks",
    "paper",
    "study",
    "using",
}
JEWEL_COLORS = [
    "#48b6ff",
    "#ff5fb7",
    "#66e28f",
    "#ffd166",
    "#9b8cff",
    "#ff7a59",
    "#3ee8d4",
    "#f55f8d",
    "#a6e35f",
    "#f0f4ff",
]
EDGE_TYPE_PRIORITY = {
    "direct_citation": 0,
    "bibliographic_coupling": 1,
    "co_citation": 2,
    "topic_similarity": 3,
    "keyword_cooccurrence": 4,
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


def get_first_author(authors: str | list[str] | tuple[str, ...] | None) -> str:
    """Return a stable first-author label from either API strings or lists."""
    if isinstance(authors, (list, tuple)):
        for author in authors:
            clean = _safe_text(author, "")
            if clean:
                return clean
        return "Unknown"
    text = _safe_text(authors, "")
    if not text:
        return "Unknown"
    first = re.split(r",|;|\bet al\.?", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return first or "Unknown"


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


def _split_search_queries(query: str) -> list[str]:
    """Accept one topic or several comma/newline/semicolon separated keyword phrases."""
    parts = re.split(r"[\n;,]+", query or "")
    queries: list[str] = []
    seen: set[str] = set()
    for part in parts:
        clean = " ".join(part.split())
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(clean)
    if not queries and query.strip():
        queries.append(" ".join(query.split()))
    return queries[:MAX_SEARCH_KEYWORDS]


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


def _render_rubric_proof() -> str:
    cards = "".join(
        '<div class="proof-card">'
        f"<strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(description)}</span>"
        "</div>"
        for title, description in RUBRIC_PROOF_POINTS
    )
    return (
        '<section class="proof-section">'
        "<h3>Backyard AI proof points</h3>"
        f'<div class="proof-grid">{cards}</div>'
        "</section>"
    )


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


def _bibtex_key(paper: PaperResult, index: int) -> str:
    first_author = re.split(r"[,;]", paper.authors)[0].strip().split()
    surname = first_author[-1] if first_author else "ref"
    surname = re.sub(r"[^A-Za-z]", "", surname).lower() or "ref"
    year = paper.year if paper.year.isdigit() else "nd"
    return f"{surname}{year}_{index}"


def _to_bibtex(papers: list[PaperResult]) -> str:
    entries = []
    for index, paper in enumerate(papers, start=1):
        fields = [f"  title = {{{paper.title}}}"]
        if paper.authors and paper.authors != "Unknown":
            fields.append(f"  author = {{{paper.authors}}}")
        if paper.year.isdigit():
            fields.append(f"  year = {{{paper.year}}}")
        if paper.doi:
            fields.append(f"  doi = {{{paper.doi}}}")
        if paper.url and paper.url != "#":
            fields.append(f"  url = {{{paper.url}}}")
        entries.append("@article{" + _bibtex_key(paper, index) + ",\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries)


def export_bibtex(results: list[PaperResult]) -> str | None:
    if not results:
        return None
    path = _ensure_export_dir() / "scholar_lens_references.bib"
    path.write_text(_to_bibtex(results), encoding="utf-8")
    return str(path)


def _paper_to_record(paper: PaperResult) -> dict[str, str]:
    return {
        "title": paper.title,
        "year": paper.year,
        "source": paper.source,
        "authors": paper.authors,
        "first_author": get_first_author(paper.authors),
        "citations": paper.citations,
        "url": paper.url,
        "doi": paper.doi,
        "abstract": paper.abstract,
    }


def _keyword_tokens(*texts: str, limit: int = 10) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", (text or "").lower()):
            token = token.strip("-")
            if token and token not in GRAPH_STOPWORDS:
                counts[token] += 1
    return [token for token, _ in counts.most_common(limit)]


def _community_label(keywords: list[str]) -> str:
    if not keywords:
        return "Unlabeled Topic"
    cleaned = [keyword.replace("-", " ").title() for keyword in keywords[:2]]
    return " / ".join(cleaned)


def _openalex_topic_labels(work: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    primary_topic = work.get("primary_topic") or {}
    if primary_topic.get("display_name"):
        labels.append(primary_topic["display_name"])
    for field_name in ("topics", "keywords"):
        for item in work.get(field_name, []) or []:
            label = item.get("display_name")
            if label and label not in labels:
                labels.append(label)
    return labels[:8]


def _topic_similarity_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[str]] = {}
    for node in nodes:
        for label in node.get("topic_labels", [])[:5]:
            buckets.setdefault(label, []).append(node["id"])

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for label, ids in buckets.items():
        if len(ids) < 2 or len(ids) > 50:
            continue
        for source, target in itertools.combinations(sorted(ids), 2):
            key = (source, target)
            if key not in edges:
                edges[key] = {
                    "source": source,
                    "target": target,
                    "weight": 0,
                    "type": "topic_similarity",
                    "label": label,
                }
            edges[key]["weight"] += 1
    return _top_weighted_edges(edges)


def _assign_graph_communities(nodes: list[dict[str, Any]]) -> None:
    label_to_id: dict[str, int] = {}
    for node in nodes:
        label = _safe_text(node.get("topic_label"), "")
        if not label:
            label = _community_label(node.get("keywords", []))
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)
        community = label_to_id[label]
        node["community"] = community
        node["community_label"] = label
        node["color"] = JEWEL_COLORS[community % len(JEWEL_COLORS)]


def _top_weighted_edges(edges: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        edges.values(),
        key=lambda edge: (
            EDGE_TYPE_PRIORITY.get(edge["type"], 9),
            -edge["weight"],
        ),
    )
    return ranked[:CONSTELLATION_MAX_EDGES]


def _keyword_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[str]] = {}
    for node in nodes:
        for keyword in node.get("keywords", [])[:8]:
            buckets.setdefault(keyword, []).append(node["id"])

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for keyword, ids in buckets.items():
        if len(ids) < 2 or len(ids) > 45:
            continue
        for source, target in itertools.combinations(sorted(ids), 2):
            key = (source, target)
            if key not in edges:
                edges[key] = {
                    "source": source,
                    "target": target,
                    "weight": 0,
                    "type": "keyword_cooccurrence",
                    "label": keyword,
                }
            edges[key]["weight"] += 1
    return _top_weighted_edges(edges)


def _constellation_payload(
    *,
    query: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    source: str,
    direct_edges: int,
    bibliographic_edges: int,
    co_citation_edges: int,
    topic_similarity_edges: int,
    fallback_used: bool,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    _assign_graph_communities(nodes)
    degrees = Counter()
    for edge in edges:
        degrees[edge["source"]] += 1
        degrees[edge["target"]] += 1
    for node in nodes:
        node["degree"] = degrees[node["id"]]

    community_labels_by_id: dict[int, str] = {}
    community_counts: Counter[int] = Counter()
    for node in nodes:
        community_id = int(node["community"])
        community_labels_by_id[community_id] = node["community_label"]
        community_counts[community_id] += 1
    return {
        "query": query,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "nodes": nodes,
        "edges": edges,
        "communities": [
            {
                "id": community_id,
                "label": community_labels_by_id[community_id],
                "count": count,
                "color": JEWEL_COLORS[community_id % len(JEWEL_COLORS)],
            }
            for community_id, count in community_counts.most_common()
        ],
        "data_completeness": {
            "paper_count": len(nodes),
            "edge_count": len(edges),
            "direct_citation_edges": direct_edges,
            "bibliographic_coupling_edges": bibliographic_edges,
            "co_citation_edges": co_citation_edges,
            "topic_similarity_edges": topic_similarity_edges,
            "keyword_fallback_used": fallback_used,
            "edge_policy": (
                "OpenAlex citation, shared-reference, and co-citation relationships"
                if not fallback_used
                else "Topic/keyword similarity fallback because retrieved citation links were sparse"
            ),
            "max_edges": CONSTELLATION_MAX_EDGES,
            "warnings": warnings or [],
        },
    }


def build_constellation_from_papers(
    query: str,
    papers: list[PaperResult],
) -> dict[str, Any]:
    nodes = []
    for index, paper in enumerate(papers):
        nodes.append(
            {
                "id": f"paper-{index}",
                "title": paper.title,
                "year": paper.year,
                "authors": paper.authors,
                "first_author": get_first_author(paper.authors),
                "citations": paper.citations,
                "url": paper.url,
                "doi": paper.doi,
                "abstract": paper.abstract,
                "topic_label": "",
                "topic_labels": [],
                "keywords": _keyword_tokens(paper.title, paper.abstract, limit=10),
            }
        )
    edges = _keyword_edges(nodes)
    return _constellation_payload(
        query=query,
        nodes=nodes,
        edges=edges,
        source="Current search results",
        direct_edges=0,
        bibliographic_edges=0,
        co_citation_edges=0,
        topic_similarity_edges=0,
        fallback_used=True,
        warnings=["Search-result records do not include reference lists, so keyword similarity is used."],
    )


def fetch_openalex_constellation(
    query: str,
    limit: int = CONSTELLATION_OPENALEX_LIMIT,
) -> tuple[dict[str, Any] | None, str | None]:
    clean_query = (query or "").strip()
    if not clean_query:
        return None, "Enter a topic for the constellation."

    params = {
        "search": clean_query,
        "per_page": min(max(limit, 25), 200),
        "filter": "has_abstract:true",
        "mailto": CONTACT_EMAIL,
        "select": (
            "id,doi,title,publication_year,authorships,cited_by_count,"
            "referenced_works,abstract_inverted_index,primary_location,primary_topic,topics,keywords"
        ),
    }
    try:
        payload = _request_json("https://api.openalex.org/works", _openalex_params(params))
    except requests.RequestException:
        return None, "OpenAlex is unavailable right now, so the constellation could not be built."

    works = payload.get("results", [])
    nodes: list[dict[str, Any]] = []
    references_by_id: dict[str, set[str]] = {}
    for work in works:
        work_id = _safe_text(work.get("id"), "")
        if not work_id:
            continue
        authors = [
            authorship.get("author", {}).get("display_name", "")
            for authorship in work.get("authorships", [])
        ]
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        topic_labels = _openalex_topic_labels(work)
        location = work.get("primary_location") or {}
        landing_page = _safe_text(
            work.get("doi")
            or location.get("landing_page_url")
            or location.get("pdf_url")
            or work_id,
            "#",
        )
        references = {
            reference
            for reference in work.get("referenced_works", [])
            if isinstance(reference, str)
        }
        references_by_id[work_id] = references
        nodes.append(
            {
                "id": work_id,
                "title": _safe_text(work.get("title"), "Untitled paper"),
                "year": _safe_text(work.get("publication_year")),
                "authors": _shorten_authors(authors),
                "first_author": get_first_author(authors),
                "citations": _safe_text(work.get("cited_by_count"), "0"),
                "url": landing_page,
                "doi": _normalize_doi(work.get("doi", "")),
                "abstract": abstract,
                "topic_label": topic_labels[0] if topic_labels else "",
                "topic_labels": topic_labels,
                "keywords": _keyword_tokens(" ".join(topic_labels), work.get("title", ""), abstract, limit=12),
            }
        )

    id_set = {node["id"] for node in nodes}
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    direct_edges = 0
    for source, references in references_by_id.items():
        for target in references & id_set:
            if source == target:
                continue
            key = tuple(sorted((source, target)))
            edge_map[key] = {
                "source": source,
                "target": target,
                "weight": 3,
                "type": "direct_citation",
                "label": "direct citation",
            }
            direct_edges += 1

    bibliographic_edges = 0
    shared_reference_pairs: Counter[tuple[str, str]] = Counter()
    reference_buckets: dict[str, list[str]] = {}
    for work_id, references in references_by_id.items():
        for reference in references:
            reference_buckets.setdefault(reference, []).append(work_id)
    for cited_work, citing_ids in reference_buckets.items():
        if len(citing_ids) < 2 or len(citing_ids) > 60:
            continue
        for source, target in itertools.combinations(sorted(citing_ids), 2):
            shared_reference_pairs[(source, target)] += 1
    for (source, target), weight in shared_reference_pairs.items():
        key = tuple(sorted((source, target)))
        if key in edge_map or weight < 2:
            continue
        edge_map[key] = {
            "source": source,
            "target": target,
            "weight": weight,
            "type": "bibliographic_coupling",
            "label": f"{weight} shared references",
        }
        bibliographic_edges += 1

    co_cited_edges = 0
    co_citation_pairs: Counter[tuple[str, str]] = Counter()
    for references in references_by_id.values():
        internal_refs = sorted(references & id_set)
        for left, right in itertools.combinations(internal_refs, 2):
            co_citation_pairs[(left, right)] += 1
    for (source, target), weight in co_citation_pairs.items():
        key = tuple(sorted((source, target)))
        if key in edge_map:
            continue
        edge_map[key] = {
            "source": source,
            "target": target,
            "weight": weight,
            "type": "co_citation",
            "label": f"co-cited {weight}x",
        }
        co_cited_edges += 1

    evidence_edges = direct_edges + bibliographic_edges + co_cited_edges
    fallback_used = evidence_edges < max(8, len(nodes) // 18)
    if fallback_used:
        edges = _topic_similarity_edges(nodes) or _keyword_edges(nodes)
    else:
        edges = _top_weighted_edges(edge_map)

    return (
        _constellation_payload(
            query=clean_query,
            nodes=nodes,
            edges=edges,
            source="OpenAlex",
            direct_edges=direct_edges,
            bibliographic_edges=bibliographic_edges,
            co_citation_edges=co_cited_edges,
            topic_similarity_edges=len(edges) if fallback_used else 0,
            fallback_used=fallback_used,
            warnings=[] if not fallback_used else ["Citation links among retrieved papers were sparse."],
        ),
        None,
    )


def export_corpus_zip(graph: dict[str, Any] | None) -> str | None:
    if not graph or not graph.get("nodes"):
        return None
    path = _ensure_export_dir() / "scholar_lens_constellation_corpus.zip"
    records = [
        {
            key: node.get(key, "")
            for key in (
                "id",
                "title",
                "year",
                "authors",
                "citations",
                "url",
                "doi",
                "abstract",
                "topic_label",
                "topic_labels",
                "keywords",
            )
        }
        for node in graph.get("nodes", [])
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("graph.json", json.dumps(graph, ensure_ascii=False, indent=2))
        archive.writestr("corpus.json", json.dumps(records, ensure_ascii=False, indent=2))
        archive.writestr("data-completeness.json", json.dumps(graph.get("data_completeness", {}), indent=2))
        archive.writestr(
            "README.txt",
            "Scholar Lens citation constellation corpus.\n"
            "Direct/co-citation edges are derived from OpenAlex referenced_works when available.\n"
            "If keyword_fallback_used is true, edges are keyword co-occurrence and not citation claims.\n",
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


def _openalex_params(params: dict[str, Any]) -> dict[str, Any]:
    request_params = dict(params)
    if OPENALEX_API_KEY:
        request_params["api_key"] = OPENALEX_API_KEY
    return request_params


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


def search_semantic_scholar(query: str) -> tuple[list[PaperResult], str | None]:
    # Requires SEMANTIC_SCHOLAR_API_KEY to be reliable; skip cleanly if absent.
    if not SEMANTIC_SCHOLAR_API_KEY:
        return [], None
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": SEARCH_LIMIT_PER_SOURCE,
        "fields": "title,year,authors,citationCount,url,abstract,externalIds",
    }
    headers = {"Accept": "application/json", "x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    try:
        response = _get_with_retries(url, params, headers=headers)
        payload = response.json()
    except (requests.RequestException, ValueError):
        return [], "Semantic Scholar is unavailable right now."

    results: list[PaperResult] = []
    for paper in payload.get("data") or []:
        authors = [author.get("name", "") for author in paper.get("authors", [])]
        external_ids = paper.get("externalIds") or {}
        results.append(
            PaperResult(
                title=_safe_text(paper.get("title"), "Untitled paper"),
                year=_safe_text(paper.get("year")),
                source="Semantic Scholar",
                authors=_shorten_authors(authors),
                citations=_safe_text(paper.get("citationCount"), "0"),
                url=_safe_text(paper.get("url"), "#"),
                abstract=_safe_text(paper.get("abstract"), ""),
                doi=_normalize_doi(external_ids.get("DOI", "")),
            )
        )
    return results, None


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
        "Semantic Scholar": ("semantic", "🧠"),
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
                    class="result-row"
                >
                    <td class="num-cell">{number}</td>
                    <td class="title-cell">{html.escape(result.title)}</td>
                    <td class="year-cell">{html.escape(result.year)}</td>
                    <td>{_source_badge(result.source)}</td>
                    <td class="authors-cell">{html.escape(result.authors)}</td>
                    <td class="citation-cell">{html.escape(result.citations)}</td>
                    <td class="action-cell"><a class="paper-link" href="{safe_url}" target="_blank" rel="noopener noreferrer">Open ↗</a></td>
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


def _collect_single_query_results(query: str) -> tuple[list[PaperResult], list[str]]:
    """Query every source in parallel for one keyword phrase."""
    search_functions = [
        search_semantic_scholar,
        search_openalex,
        search_crossref,
        search_arxiv,
        search_pubmed,
    ]
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


def _collect_results(query: str) -> tuple[list[PaperResult], list[str]]:
    """Query one or more keyword phrases, then de-duplicate and rank results."""
    queries = _split_search_queries(query)
    if not queries:
        return [], []
    if len(queries) == 1:
        return _collect_single_query_results(queries[0])

    results: list[PaperResult] = []
    warnings: list[str] = []
    for search_query in queries:
        source_results, source_warnings = _collect_single_query_results(search_query)
        results.extend(source_results)
        warnings.extend(source_warnings)

    combined_query = " ".join(queries)
    return _rank_results(_dedupe_results(results), combined_query), warnings


def _selector_choices(results: list[PaperResult]) -> list[str]:
    return [f"{index + 1}. {result.title}" for index, result in enumerate(results)]


def _compare_selector_updates(results: list[PaperResult]) -> tuple[gr.Dropdown, gr.Dropdown]:
    choices = _selector_choices(results)
    return (
        gr.update(choices=choices, value=choices[0] if len(results) >= 1 else None),
        gr.update(choices=choices, value=choices[1] if len(results) >= 2 else None),
    )


def search_all_sources(query: str):
    clean_query = query.strip()
    search_queries = _split_search_queries(clean_query)
    if not clean_query:
        table, label, page = _page_view([], 0)
        prev_update, next_update = _pagination_updates([], page)
        compare_left, compare_right = _compare_selector_updates([])
        return (
            "Enter one research topic, or multiple keyword phrases separated by commas or new lines.",
            table,
            "",
            [],
            gr.update(choices=[], value=None),
            label,
            page,
            prev_update,
            next_update,
            None,
            compare_left,
            compare_right,
        )
    if len(clean_query) > SEARCH_QUERY_CHAR_LIMIT:
        table, label, page = _page_view([], 0)
        prev_update, next_update = _pagination_updates([], page)
        compare_left, compare_right = _compare_selector_updates([])
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
            compare_left,
            compare_right,
        )

    results, warnings = _collect_results(clean_query)
    query_note = (
        f" across **{len(search_queries)}** keyword searches"
        if len(search_queries) > 1
        else " across all sources"
    )

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

    if results:
        status = f"Found **{len(results)}** papers{query_note}."
        if warnings:
            status += " " + " ".join(warnings)

    table, label, page = _page_view(results, 0)
    prev_update, next_update = _pagination_updates(results, page)
    results_csv = export_results_csv(results) if results else None
    compare_left, compare_right = _compare_selector_updates(results)
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
        compare_left,
        compare_right,
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


def load_selected_paper_reset_chat(
    selected_index: Any,
    results: list[PaperResult],
) -> tuple[str, str, str, gr.Tabs, str, str, str]:
    try:
        index = int(selected_index)
    except (TypeError, ValueError):
        index = None
    paper_text, load_status, summary, tab_update = load_selected_paper(index, results)
    if paper_text and not summary:
        summary = "Paper loaded. Click Summarize with AI to generate the summary."
    return paper_text, load_status, summary, tab_update, "", "", DEFAULT_PAPER_CHAT_ANSWER


def summarize_row_selection(
    selected_index: Any,
    results: list[PaperResult],
) -> tuple[str, str, str, gr.Tabs, gr.Dropdown, str, str, str]:
    paper_text, load_status, summary, tab_update, results_text, chat_question, chat_output = load_selected_paper_reset_chat(
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
        results_text,
        chat_question,
        chat_output,
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


def _empty_constellation_html(message: str = "Build a constellation to explore connected papers.") -> str:
    safe_message = html.escape(message)
    return f"""
    <div class="constellation-empty">
        <h3>Literature Constellation</h3>
        <p>{safe_message}</p>
    </div>
    """


def _render_constellation_html(graph: dict[str, Any] | None) -> str:
    if not graph or not graph.get("nodes"):
        return _empty_constellation_html()

    frame_id = f"constellation-{uuid.uuid4().hex}"
    graph_json = json.dumps(graph, ensure_ascii=False)
    srcdoc = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #02030a; color: #f8fafc; font-family: Inter, Arial, sans-serif; }}
#app {{ position: relative; width: 100vw; height: 100vh; overflow: hidden; background: radial-gradient(circle at 48% 42%, rgba(42, 75, 140, .22), transparent 34%), linear-gradient(135deg, #02030a 0%, #070816 58%, #02030a 100%); }}
canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }}
canvas.dragging {{ cursor: grabbing; }}
.top {{ position: absolute; top: 22px; left: 28px; right: 28px; pointer-events: none; z-index: 4; display: flex; justify-content: space-between; gap: 24px; align-items: start; }}
.brand {{ color: rgba(255,255,255,.46); font: 800 10px/1.4 monospace; letter-spacing: 3px; text-transform: uppercase; }}
h1 {{ margin: 5px 0 6px; font: italic 700 clamp(28px, 4vw, 52px)/1 Georgia, serif; color: rgba(255,255,255,.94); text-shadow: 0 0 28px rgba(72,182,255,.22); }}
.meta {{ color: rgba(255,255,255,.58); font: 700 12px/1.45 monospace; letter-spacing: .7px; max-width: 680px; }}
.controls {{ pointer-events: auto; display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; max-width: 380px; }}
.chip {{ border: 1px solid rgba(255,255,255,.12); color: rgba(255,255,255,.74); background: rgba(5,7,18,.62); border-radius: 999px; padding: 8px 11px; font: 800 10px/1 monospace; letter-spacing: 1px; backdrop-filter: blur(12px); }}
.chip.active {{ color: #06111f; background: #48b6ff; border-color: rgba(72,182,255,.8); box-shadow: 0 0 22px rgba(72,182,255,.32); }}
.legend {{ position: absolute; right: 24px; bottom: 26px; width: min(330px, 30vw); max-height: 38vh; overflow: auto; padding: 14px 16px; background: rgba(5,7,18,.72); border: 1px solid rgba(255,255,255,.12); border-radius: 8px; backdrop-filter: blur(16px); z-index: 6; }}
.legend-title {{ color: rgba(255,255,255,.44); font: 800 10px/1.5 monospace; letter-spacing: 2px; margin-bottom: 8px; }}
.legend-row {{ display: grid; grid-template-columns: 13px 1fr auto; gap: 9px; align-items: center; padding: 6px 0; cursor: pointer; color: rgba(255,255,255,.72); font: 700 11px/1.25 monospace; }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; box-shadow: 0 0 16px currentColor; }}
.legend-row:hover, .legend-row.active {{ color: #fff; }}
.detail {{ position: absolute; top: 132px; right: 24px; width: min(420px, 32vw); max-height: 54vh; overflow: auto; padding: 17px 19px; background: rgba(5,7,18,.88); border: 1px solid rgba(72,182,255,.28); border-radius: 8px; box-shadow: 0 0 34px rgba(72,182,255,.12); z-index: 7; display: none; }}
.detail h2 {{ margin: 0 0 8px; font: 700 18px/1.22 Georgia, serif; color: #fff; }}
.detail .byline {{ color: rgba(255,255,255,.58); font: 700 11px/1.55 monospace; margin-bottom: 12px; }}
.detail .relation {{ color: #66e28f; font: 800 10px/1.4 monospace; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }}
.detail p {{ color: rgba(255,255,255,.78); font: 13px/1.55 Arial, sans-serif; }}
.detail a {{ display: inline-flex; margin-top: 10px; color: #48b6ff; text-decoration: none; font-weight: 800; }}
.evidence {{ position: absolute; left: 50%; bottom: 20px; transform: translateX(-50%); max-width: min(720px, 72vw); padding: 9px 13px; border-radius: 999px; background: rgba(5,7,18,.66); border: 1px solid rgba(255,255,255,.12); color: rgba(255,255,255,.62); font: 800 10px/1.35 monospace; letter-spacing: 1px; text-align: center; backdrop-filter: blur(14px); z-index: 5; pointer-events: none; }}
@media (max-width: 820px) {{ .top {{ left: 14px; right: 14px; display: block; }} .controls {{ margin-top: 8px; justify-content: flex-start; }} .legend {{ left: 14px; right: 14px; bottom: 14px; width: auto; max-height: 26vh; }} .detail {{ left: 14px; right: 14px; width: auto; top: 156px; max-height: 38vh; }} .evidence {{ display: none; }} }}
</style>
</head>
<body>
<div id="app">
  <canvas id="map"></canvas>
  <div class="top">
    <div>
      <div class="brand">CONNECTED LITERATURE MAP</div>
      <h1>Literature Constellation</h1>
      <div class="meta" id="meta"></div>
    </div>
    <div class="controls" id="controls"></div>
  </div>
  <div class="legend" id="legend"><div class="legend-title">CLUSTERS - HOVER TO ISOLATE</div></div>
  <div class="detail" id="detail"></div>
  <div class="evidence" id="evidence"></div>
</div>
<script>
const graph = {graph_json};
function escapeHtml(value) {{ return String(value || '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[ch])); }}
function escapeAttr(value) {{ return escapeHtml(value).replace(/`/g, '&#096;'); }}
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const nodeById = new Map();
const neighbors = new Map();
let activeCommunity = null;
let selectedId = null;
let hoverId = null;
let transform = {{ x: 0, y: 0, scale: 1 }};
let dragStart = null;
const relationLabels = {{
  direct_citation: 'Citation',
  bibliographic_coupling: 'Shared refs',
  co_citation: 'Co-cited',
  topic_similarity: 'Topic',
  keyword_cooccurrence: 'Keyword'
}};
const relationColors = {{
  direct_citation: '#ffffff',
  bibliographic_coupling: '#48b6ff',
  co_citation: '#66e28f',
  topic_similarity: '#ffd166',
  keyword_cooccurrence: '#9b8cff'
}};
let activeRelations = new Set(Object.keys(relationLabels));
graph.nodes.forEach((node, index) => {{
  nodeById.set(node.id, node);
  neighbors.set(node.id, new Set());
  const communityIndex = Math.max(0, graph.communities.findIndex(comm => comm.id === node.community));
  const angle = (communityIndex / Math.max(1, graph.communities.length)) * Math.PI * 2;
  const ring = 210 + communityIndex * 9;
  const jitter = ((index * 73) % 29) - 14;
  node.x = Math.cos(angle) * ring + Math.cos(index * 1.618) * jitter * 4;
  node.y = Math.sin(angle) * ring + Math.sin(index * 1.618) * jitter * 4;
  node.vx = 0;
  node.vy = 0;
  node.radius = 5 + Math.min(18, Math.sqrt(Math.max(0, Number(node.citations) || 0)) * .65) + Math.min(8, Math.sqrt((node.degree || 0) + 1));
}});
graph.edges.forEach(edge => {{
  if (neighbors.has(edge.source)) neighbors.get(edge.source).add(edge.target);
  if (neighbors.has(edge.target)) neighbors.get(edge.target).add(edge.source);
}});
function edgeDistance(type) {{
  return type === 'direct_citation' ? 82 : type === 'bibliographic_coupling' ? 112 : type === 'co_citation' ? 126 : type === 'topic_similarity' ? 148 : 162;
}}
function runLayout() {{
  const nodes = graph.nodes;
  for (let step = 0; step < 260; step++) {{
    for (let i = 0; i < nodes.length; i++) {{
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {{
        const b = nodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dist2 = dx * dx + dy * dy + .01;
        let force = 1550 / dist2;
        let dist = Math.sqrt(dist2);
        let fx = (dx / dist) * force;
        let fy = (dy / dist) * force;
        a.vx -= fx; a.vy -= fy; b.vx += fx; b.vy += fy;
      }}
    }}
    graph.edges.forEach(edge => {{
      const a = nodeById.get(edge.source), b = nodeById.get(edge.target);
      if (!a || !b) return;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      let dist = Math.sqrt(dx * dx + dy * dy) || 1;
      let wanted = edgeDistance(edge.type) - Math.min(36, Math.log1p(edge.weight || 1) * 9);
      let force = (dist - wanted) * (.006 + Math.min(.018, (edge.weight || 1) * .0015));
      let fx = (dx / dist) * force;
      let fy = (dy / dist) * force;
      a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
    }});
    nodes.forEach(node => {{
      const angle = (node.community / Math.max(1, graph.communities.length)) * Math.PI * 2;
      const anchorX = Math.cos(angle) * 260;
      const anchorY = Math.sin(angle) * 190;
      node.vx += (anchorX - node.x) * .002;
      node.vy += (anchorY - node.y) * .002;
      node.x += node.vx;
      node.y += node.vy;
      node.vx *= .72;
      node.vy *= .72;
    }});
  }}
}}
runLayout();
function resize() {{
  canvas.width = Math.floor(innerWidth * devicePixelRatio);
  canvas.height = Math.floor(innerHeight * devicePixelRatio);
  canvas.style.width = innerWidth + 'px';
  canvas.style.height = innerHeight + 'px';
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  draw();
}}
function toScreen(node) {{ return {{ x: innerWidth / 2 + (node.x + transform.x) * transform.scale, y: innerHeight / 2 + (node.y + transform.y) * transform.scale }}; }}
function fromScreen(x, y) {{ return {{ x: (x - innerWidth / 2) / transform.scale - transform.x, y: (y - innerHeight / 2) / transform.scale - transform.y }}; }}
function visibleNode(node) {{
  const selectedNeighbors = selectedId ? neighbors.get(selectedId) || new Set() : new Set();
  return (activeCommunity === null || node.community === activeCommunity) && (!selectedId || node.id === selectedId || selectedNeighbors.has(node.id));
}}
function edgeVisible(edge) {{
  if (!activeRelations.has(edge.type)) return false;
  const a = nodeById.get(edge.source), b = nodeById.get(edge.target);
  return a && b && visibleNode(a) && visibleNode(b);
}}
function draw() {{
  ctx.clearRect(0, 0, innerWidth, innerHeight);
  const grad = ctx.createRadialGradient(innerWidth * .46, innerHeight * .46, 20, innerWidth * .46, innerHeight * .46, Math.max(innerWidth, innerHeight) * .62);
  grad.addColorStop(0, 'rgba(39,69,132,.28)');
  grad.addColorStop(1, 'rgba(2,3,10,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, innerWidth, innerHeight);
  graph.edges.forEach(edge => {{
    if (!edgeVisible(edge)) return;
    const a = toScreen(nodeById.get(edge.source));
    const b = toScreen(nodeById.get(edge.target));
    const selected = selectedId && (edge.source === selectedId || edge.target === selectedId);
    ctx.strokeStyle = relationColors[edge.type] || 'rgba(255,255,255,.35)';
    ctx.globalAlpha = selected ? .68 : edge.type === 'direct_citation' ? .34 : .16;
    ctx.lineWidth = selected ? 2.4 : Math.max(.55, Math.min(2.4, Math.log1p(edge.weight || 1) * .46));
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }});
  ctx.globalAlpha = 1;
  graph.nodes.forEach(node => {{
    const p = toScreen(node);
    const selected = node.id === selectedId;
    const hovered = node.id === hoverId;
    const neighbor = selectedId && (neighbors.get(selectedId) || new Set()).has(node.id);
    const visible = visibleNode(node);
    const r = (node.radius * (selected ? 1.45 : hovered ? 1.3 : neighbor ? 1.16 : 1)) * transform.scale;
    ctx.globalAlpha = visible ? 1 : .14;
    ctx.fillStyle = node.color || '#48b6ff';
    ctx.shadowColor = node.color || '#48b6ff';
    ctx.shadowBlur = visible ? (selected || hovered ? 24 : 13) : 0;
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(3, r), 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = selected || hovered ? '#fff' : 'rgba(255,255,255,.26)';
    ctx.lineWidth = selected || hovered ? 2 : 1;
    ctx.stroke();
  }});
  ctx.globalAlpha = 1;
}}
function nearestNode(event) {{
  const rect = canvas.getBoundingClientRect();
  let best = null;
  let bestDist = Infinity;
  graph.nodes.forEach(node => {{
    const p = toScreen(node);
    const dx = event.clientX - rect.left - p.x;
    const dy = event.clientY - rect.top - p.y;
    const limit = Math.max(13, node.radius * transform.scale + 6);
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < limit && dist < bestDist) {{ best = node; bestDist = dist; }}
  }});
  return best;
}}
function showDetail(node) {{
  selectedId = node.id;
  const detail = document.getElementById('detail');
  detail.style.display = 'block';
  const connected = Array.from(neighbors.get(node.id) || []).length;
  detail.innerHTML = `<div class="relation">${{escapeHtml(node.community_label || 'Topic cluster')}} - ${{connected}} nearby papers</div><h2>${{escapeHtml(node.title)}}</h2><div class="byline">${{escapeHtml(node.first_author || node.authors)}} - ${{escapeHtml(node.year)}} - ${{escapeHtml(String(node.citations))}} cites</div><p>${{escapeHtml(node.abstract || 'No abstract was available for this paper.')}}</p><a href="${{escapeAttr(node.url || '#')}}" target="_blank" rel="noopener noreferrer">Open paper</a>`;
  draw();
}}
canvas.addEventListener('mousemove', event => {{
  if (dragStart) {{
    transform.x = dragStart.originX + (event.clientX - dragStart.x) / transform.scale;
    transform.y = dragStart.originY + (event.clientY - dragStart.y) / transform.scale;
    draw();
    return;
  }}
  const node = nearestNode(event);
  const nextHover = node ? node.id : null;
  if (nextHover !== hoverId) {{ hoverId = nextHover; draw(); }}
}});
canvas.addEventListener('mousedown', event => {{
  dragStart = {{ x: event.clientX, y: event.clientY, originX: transform.x, originY: transform.y }};
  canvas.classList.add('dragging');
}});
addEventListener('mouseup', () => {{ dragStart = null; canvas.classList.remove('dragging'); }});
canvas.addEventListener('click', event => {{
  if (dragStart) return;
  const node = nearestNode(event);
  if (node) showDetail(node);
}});
canvas.addEventListener('wheel', event => {{
  event.preventDefault();
  const point = fromScreen(event.offsetX, event.offsetY);
  const factor = event.deltaY < 0 ? 1.12 : .9;
  transform.scale = Math.max(.45, Math.min(2.8, transform.scale * factor));
  const after = fromScreen(event.offsetX, event.offsetY);
  transform.x += after.x - point.x;
  transform.y += after.y - point.y;
  draw();
}}, {{ passive: false }});
document.getElementById('meta').textContent = `${{graph.nodes.length}} papers - ${{graph.communities.length}} clusters - ${{graph.edges.length}} relationships - ${{graph.query}}`;
document.getElementById('evidence').textContent = graph.data_completeness.edge_policy;
const controlsEl = document.getElementById('controls');
Object.entries(relationLabels).forEach(([type, label]) => {{
  const count = graph.edges.filter(edge => edge.type === type).length;
  if (!count) return;
  const button = document.createElement('button');
  button.className = 'chip active';
  button.type = 'button';
  button.textContent = `${{label}} ${{count}}`;
  button.addEventListener('click', () => {{
    if (activeRelations.has(type)) {{ activeRelations.delete(type); button.classList.remove('active'); }}
    else {{ activeRelations.add(type); button.classList.add('active'); }}
    draw();
  }});
  controlsEl.appendChild(button);
}});
const legend = document.getElementById('legend');
graph.communities.forEach(comm => {{
  const row = document.createElement('div');
  row.className = 'legend-row';
  row.innerHTML = `<span class="dot" style="color:${{comm.color}}; background:${{comm.color}}"></span><span>${{escapeHtml(comm.label)}}</span><span>${{comm.count}}</span>`;
  row.addEventListener('mouseenter', () => {{ activeCommunity = comm.id; draw(); row.classList.add('active'); }});
  row.addEventListener('mouseleave', () => {{ activeCommunity = null; draw(); row.classList.remove('active'); }});
  row.addEventListener('click', () => {{ activeCommunity = activeCommunity === comm.id ? null : comm.id; draw(); }});
  legend.appendChild(row);
}});
addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""
    return (
        f'<iframe id="{frame_id}" class="constellation-frame" '
        f'sandbox="allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox" '
        f'srcdoc="{html.escape(srcdoc, quote=True)}"></iframe>'
    )


def build_openalex_constellation(query: str) -> tuple[str, str, dict[str, Any] | None, str | None]:
    graph, error = fetch_openalex_constellation(query)
    if error:
        return error, _empty_constellation_html(error), None, None
    assert graph is not None
    zip_path = export_corpus_zip(graph)
    completeness = graph["data_completeness"]
    status = (
        f"Built {completeness['paper_count']} papers and {completeness['edge_count']} edges. "
        f"Data completeness: {completeness['edge_policy']}."
    )
    return status, _render_constellation_html(graph), graph, zip_path


def build_search_result_constellation(
    query: str,
    results: list[PaperResult],
) -> tuple[str, str, dict[str, Any] | None, str | None]:
    if not results:
        message = "Search first, then build a constellation from the current results."
        return message, _empty_constellation_html(message), None, None
    graph = build_constellation_from_papers(query or "current search results", results)
    zip_path = export_corpus_zip(graph)
    return (
        f"Built a keyword fallback constellation from {len(results)} current search results.",
        _render_constellation_html(graph),
        graph,
        zip_path,
    )


def compare_papers_with_ai(
    left_index: Any,
    right_index: Any,
    results: list[PaperResult],
) -> str:
    try:
        left = int(left_index)
        right = int(right_index)
    except (TypeError, ValueError):
        return "Choose two papers from the current search results."
    if left == right:
        return "Choose two different papers to compare."
    if left < 0 or right < 0 or left >= len(results) or right >= len(results):
        return "One of the selected papers is no longer available in the search results."

    papers = [results[left], results[right]]
    contexts = []
    for index, paper in enumerate(papers, start=1):
        abstract = _trim_to_budget(paper.abstract, MAX_ABSTRACT_CHARS, MAX_ABSTRACT_TOKENS)
        contexts.append(
            f"[{index}] Title: {paper.title}\n"
            f"Authors: {paper.authors}\n"
            f"Year: {paper.year}\n"
            f"Source: {paper.source}\n"
            f"Citations: {paper.citations}\n"
            f"URL: {paper.url}\n"
            f"Abstract: {abstract or 'No abstract available.'}"
        )
    prompt = (
        f"Compare these two papers for a researcher using {MODEL_DISPLAY_NAME}. "
        "Use only the provided metadata and abstracts. Cover: shared problem, "
        "method/data differences, claims, limitations, which one to read first "
        "for which purpose, and open questions. If evidence is missing, say so."
    )
    return synthesize_with_modal(prompt, "\n\n".join(contexts))


def clear_search():
    table, label, page = _page_view([], 0)
    prev_update, next_update = _pagination_updates([], page)
    compare_left, compare_right = _compare_selector_updates([])
    graph_message = _empty_constellation_html()
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
        compare_left,
        compare_right,
        "",
        DEFAULT_COMPARE_ANSWER,
        graph_message,
        None,
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
.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container [role="tabpanel"],
.gradio-container .panel,
.gradio-container .block,
.gradio-container .form,
.gradio-container .contain,
.gradio-container .wrap:not(:has(input.border-none)),
.gradio-container .wrap-inner:not(:has(input.border-none)) {
    background: var(--sl-bg) !important;
    color: var(--sl-text) !important;
    border-color: var(--sl-border) !important;
}
.tab-nav {
    border-bottom: 1px solid var(--sl-border) !important;
    background: var(--sl-panel) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 4px !important;
}
.gradio-container [role="tab"],
.gradio-container .tab-nav button {
    background: transparent !important;
    color: var(--sl-muted) !important;
    border-color: transparent !important;
    box-shadow: none !important;
    min-height: 36px !important;
}
/* include :active and :focus-visible so a clicked tab never flashes white */
.gradio-container [role="tab"]:hover,
.gradio-container [role="tab"]:focus,
.gradio-container [role="tab"]:focus-visible,
.gradio-container [role="tab"]:active,
.gradio-container .tab-nav button:hover,
.gradio-container .tab-nav button:focus,
.gradio-container .tab-nav button:focus-visible,
.gradio-container .tab-nav button:active {
    background: rgba(59, 130, 246, 0.18) !important;
    color: var(--sl-text-bright) !important;
    border-color: rgba(96, 165, 250, 0.55) !important;
    box-shadow: inset 0 -2px 0 var(--sl-accent) !important;
}
.gradio-container [role="tab"][aria-selected="true"],
.gradio-container [role="tab"][aria-selected="true"]:active,
.tab-nav button.selected {
    color: var(--sl-accent-bright) !important;
    border-bottom: 2px solid var(--sl-accent) !important;
    background: var(--sl-accent-soft) !important;
    box-shadow: inset 0 -2px 0 var(--sl-accent) !important;
}

.gradio-container button[disabled],
.gradio-container button:disabled {
    background: rgba(30, 41, 59, 0.7) !important;
    color: rgba(148, 163, 184, 0.7) !important;
    border-color: rgba(51, 65, 85, 0.8) !important;
    opacity: 1 !important;
}
.gradio-container .overflow-menu button {
    color: var(--sl-text-bright) !important;
}
.gradio-container .overflow-dropdown {
    background: var(--sl-panel) !important;
    border: 1px solid var(--sl-border) !important;
    box-shadow: 0 18px 45px rgba(2, 6, 23, 0.55) !important;
}
.gradio-container .overflow-dropdown button {
    background: transparent !important;
    color: var(--sl-text-bright) !important;
}
.gradio-container .overflow-dropdown button:hover,
.gradio-container .overflow-dropdown button:focus {
    background: var(--sl-accent-soft) !important;
    color: var(--sl-accent-bright) !important;
}
.gradio-container .overflow-dropdown button.selected {
    background: rgba(59, 130, 246, 0.24) !important;
    color: var(--sl-text-bright) !important;
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
.readable-select input,
.readable-select [role="listbox"] input,
.readable-select .secondary-wrap input,
.gradio-container input.border-none,
.gradio-container .secondary-wrap input {
    background: #f8fafc !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    opacity: 1 !important;
}
.readable-select input::placeholder,
.gradio-container input.border-none::placeholder,
.gradio-container .secondary-wrap input::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
}
.readable-select .wrap,
.readable-select .wrap-inner,
.readable-select .secondary-wrap,
.gradio-container .wrap:has(input.border-none),
.gradio-container .wrap-inner:has(input.border-none),
.gradio-container .secondary-wrap:has(input.border-none) {
    background: #f8fafc !important;
    color: #0f172a !important;
    border-color: rgba(96, 165, 250, 0.38) !important;
}
.readable-select label,
.readable-select span {
    color: var(--sl-muted) !important;
}
.readable-select [role="option"],
.readable-select li,
.gradio-container [role="option"],
.gradio-container li[role="option"] {
    color: #0f172a !important;
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
.gradio-container button:not(.primary):not([role="tab"]) {
    background: var(--sl-panel) !important;
    color: var(--sl-text-bright) !important;
    border: 1px solid var(--sl-border) !important;
}
.gradio-container button:not(.primary):not([role="tab"]):hover,
.gradio-container button:not(.primary):not([role="tab"]):focus {
    background: var(--sl-panel-soft) !important;
    border-color: rgba(96, 165, 250, 0.55) !important;
    color: var(--sl-text-bright) !important;
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
.result-row { transition: background 140ms ease; }
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

.source-badge.semantic { color: #a78bfa !important; background: rgba(167, 139, 250, 0.12); border: 1px solid rgba(167, 139, 250, 0.35); }
.source-badge.openalex { color: #22d3ee !important; background: rgba(34, 211, 238, 0.1); border: 1px solid rgba(34, 211, 238, 0.3); }
.source-badge.crossref { color: #fbbf24 !important; background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); }
.source-badge.arxiv { color: #f87171 !important; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); }
.source-badge.pubmed { color: #34d399 !important; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); }

.status-line {
    color: var(--sl-muted) !important;
    padding: 4px 2px;
}
.empty-state {
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 8px;
    color: var(--sl-muted) !important;
    border: 1px dashed rgba(96, 165, 250, 0.25);
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.58);
    text-align: center;
    padding: 24px;
}
.empty-state h3 {
    color: var(--sl-text-bright) !important;
    margin: 0;
}
.empty-state p {
    color: var(--sl-muted) !important;
    margin: 0;
}
.empty-crest {
    font-size: 28px;
    color: var(--sl-accent-bright);
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
.btn-crimson button {
    background: rgba(127, 29, 29, 0.86) !important;
    border-color: rgba(239, 68, 68, 0.4) !important;
    color: #fecaca !important;
    box-shadow: 0 4px 14px rgba(127, 29, 29, 0.25) !important;
}
.btn-crimson button:hover,
.btn-crimson button:focus {
    background: rgba(185, 28, 28, 0.92) !important;
    border-color: rgba(248, 113, 113, 0.7) !important;
    color: #fff !important;
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
.proof-section {
    margin-top: 18px;
    padding-top: 18px;
    border-top: 1px solid var(--sl-border-soft);
}
.proof-section h3 {
    margin: 0 0 12px;
    color: var(--sl-text-bright);
    font-size: 15px;
}
.proof-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px;
}
.proof-card {
    border: 1px solid var(--sl-border-soft);
    border-radius: 8px;
    padding: 12px;
    background: rgba(15, 23, 42, 0.38);
}
.proof-card strong {
    display: block;
    color: var(--sl-accent-bright);
    margin-bottom: 5px;
}
.proof-card span {
    color: var(--sl-muted);
    font-size: 13px;
    line-height: 1.45;
}
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

.constellation-frame {
    width: 100%;
    height: min(860px, 78vh);
    min-height: 620px;
    border: 1px solid rgba(214, 173, 79, 0.25);
    border-radius: 8px;
    background: #000;
    overflow: hidden;
}
.constellation-empty {
    min-height: 420px;
    border: 1px solid rgba(214, 173, 79, 0.22);
    border-radius: 8px;
    background:
        radial-gradient(circle at 50% 35%, rgba(214, 173, 79, 0.08), transparent 34%),
        #020617;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: var(--sl-muted);
}
.constellation-empty h3 {
    margin: 0 0 8px;
    color: var(--sl-text-bright);
    font-family: "Playfair Display", "Georgia", serif;
    font-size: 30px;
}
.constellation-empty p { max-width: 520px; margin: 0; }

/* (UI/UX polish layer reverted - it caused a layout regression) */

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
        // Use the framework's native value setter so Gradio/Svelte actually
        // registers the change, then fire `input` to trigger the .input()
        // handler. No setTimeout/second-click race.
        const setter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype, 'value'
        ).set;
        setter.call(input, String(index));
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
    }
    document.querySelectorAll('.result-row').forEach(r => r.classList.remove('selected'));
    const row = document.getElementById('row-' + index);
    if (row) row.classList.add('selected');
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
                f"""
                <header class="app-header">
                    <div class="header-row">
                        <div class="crest">🔬</div>
                        <div class="header-text">
                            <h1>Scholar <span class="accent">Lens</span></h1>
                            <p>Small-model literature review for atmospheric science</p>
                            <div class="header-chips">
                                <span class="header-chip">Semantic Scholar</span>
                                <span class="header-chip">OpenAlex</span>
                                <span class="header-chip">Crossref</span>
                                <span class="header-chip">arXiv</span>
                                <span class="header-chip">PubMed</span>
                                <span class="header-chip">{MODEL_PROVIDER_BADGE}</span>
                            </div>
                        </div>
                    </div>
                </header>
                """
            )

            papers_state = gr.State([])
            page_state = gr.State(0)
            graph_state = gr.State(None)

            with gr.Tabs(selected="ask") as app_tabs:
                with gr.Tab("💬  Ask", id="ask"):
                    gr.Markdown(
                        f"Ask a research question. Scholar Lens searches **OpenAlex, "
                        f"Crossref, arXiv, and PubMed**, then **{MODEL_DISPLAY_NAME}** "
                        f"writes a synthesized, **cited** answer grounded only in the "
                        f"retrieved abstracts.",
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

                with gr.Tab("Compare", id="compare"):
                    gr.Markdown(
                        f"Search for papers, choose any two results, and ask {MODEL_DISPLAY_NAME} to compare their methods, claims, limitations, and best use cases.",
                        elem_classes=["ask-intro"],
                    )
                    with gr.Row(elem_classes=["action-row"]):
                        compare_left_selector = gr.Dropdown(
                            label="Paper A",
                            choices=[],
                            type="index",
                            interactive=True,
                            scale=1,
                            elem_classes=["readable-select"],
                        )
                        compare_right_selector = gr.Dropdown(
                            label="Paper B",
                            choices=[],
                            type="index",
                            interactive=True,
                            scale=1,
                            elem_classes=["readable-select"],
                        )
                        compare_button = gr.Button("Compare with Nemotron", variant="primary", scale=0, min_width=190)
                    compare_output = gr.Markdown(
                        DEFAULT_COMPARE_ANSWER,
                        elem_classes=["answer-card"],
                    )
                    compare_button.click(
                        fn=compare_papers_with_ai,
                        inputs=[compare_left_selector, compare_right_selector, papers_state],
                        outputs=compare_output,
                        show_progress="full",
                    )

                with gr.Tab("🔍  Search", id="search"):
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Search topic or keywords",
                            placeholder="One topic, or several keywords separated by commas/new lines",
                            scale=5,
                            container=False,
                            lines=2,
                            max_length=SEARCH_QUERY_CHAR_LIMIT,
                        )
                        search_button = gr.Button("🔍 Search", variant="primary", scale=1)
                        with gr.Column(scale=0, min_width=100, elem_classes=["btn-crimson"]):
                            clear_button = gr.Button("Clear")

                    status_output = gr.Markdown("Ready for search.", elem_classes=["status-line"])
                    insights_output = gr.HTML()
                    results_output = gr.HTML(_render_results_table([]))
                    with gr.Row():
                        results_download = gr.DownloadButton(
                            "Download Results CSV",
                            value=None,
                            size="sm",
                            elem_classes=["download-action"],
                        )
                        bibtex_download = gr.DownloadButton(
                            "Download BibTeX",
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
                            elem_classes=["readable-select"],
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
                        compare_left_selector,
                        compare_right_selector,
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
                    bibtex_download.click(
                        fn=export_bibtex,
                        inputs=papers_state,
                        outputs=bibtex_download,
                    )

                    hidden_selected_index = gr.Textbox(
                        label="Selected row index",
                        elem_id="hidden_index_input",
                        elem_classes=["hidden-component"],
                    )

                with gr.Tab("Constellation", id="constellation"):
                    gr.Markdown(
                        "Start the demo here: build a literature constellation, pick two papers, then use Nemotron for grounded comparison.",
                        elem_classes=["ask-intro"],
                    )
                    with gr.Row():
                        graph_query = gr.Textbox(
                            label="Constellation topic",
                            value="",
                            placeholder="e.g. atmospheric rivers, satellite precipitation, aerosol-cloud interactions",
                            scale=5,
                            container=False,
                            max_length=SEARCH_QUERY_CHAR_LIMIT,
                        )
                        build_graph_button = gr.Button("Build Map", variant="primary", scale=1)
                    with gr.Row():
                        build_from_search_button = gr.Button("Use Current Search Results", size="sm")
                        corpus_download = gr.DownloadButton(
                            "Download Corpus ZIP",
                            value=None,
                            size="sm",
                            elem_classes=["download-action"],
                        )
                    graph_status = gr.Markdown(
                        "Build a constellation from OpenAlex or from current search results.",
                        elem_classes=["status-line"],
                    )
                    graph_output = gr.HTML(_empty_constellation_html())

                    build_graph_button.click(
                        fn=build_openalex_constellation,
                        inputs=graph_query,
                        outputs=[graph_status, graph_output, graph_state, corpus_download],
                        show_progress="full",
                    )
                    graph_query.submit(
                        fn=build_openalex_constellation,
                        inputs=graph_query,
                        outputs=[graph_status, graph_output, graph_state, corpus_download],
                        show_progress="full",
                    )
                    build_from_search_button.click(
                        fn=build_search_result_constellation,
                        inputs=[query_input, papers_state],
                        outputs=[graph_status, graph_output, graph_state, corpus_download],
                        show_progress="full",
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
                        fn=load_selected_paper_reset_chat,
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
                    # Row clicks set this hidden textbox via the framework-safe
                    # setter; .input() fires reliably (no setTimeout race).
                    hidden_selected_index.input(
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
                        f"""
                        <div class="about-card">
                            <h2>🎓 About Scholar Lens</h2>
                            <p>
                                Built for a real <strong>atmospheric-science professor</strong> who was
                                losing hours every week juggling four paper databases &mdash; copying
                                abstracts, losing track, re-reading the same studies. <strong>Scholar
                                Lens</strong> turns one research question into a cited, cross-database
                                answer in seconds.
                            </p>
                            <p>
                                It is a small-model academic discovery engine: the model does the
                                reading and synthesis, not just decoration.
                            </p>
                            <p>
                                It performs real-time searches across <strong>OpenAlex</strong>,
                                <strong>Crossref</strong>, <strong>arXiv</strong>, and <strong>PubMed</strong>
                                using parallel processing, then uses <strong>{MODEL_DISPLAY_NAME}</strong>,
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
                                {MODEL_PROVIDER_BADGE} &middot; model small enough for the consumer-GPU story.
                            </p>
                            {_render_rubric_proof()}
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
                    compare_left_selector,
                    compare_right_selector,
                    graph_status,
                    compare_output,
                    graph_output,
                    corpus_download,
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
