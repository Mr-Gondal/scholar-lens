import unittest
from unittest.mock import Mock

import app


def paper(
    title: str,
    *,
    year: str = "2024",
    abstract: str = "",
    citations: str = "0",
) -> app.PaperResult:
    return app.PaperResult(
        title=title,
        year=year,
        source="OpenAlex",
        authors="A. Author",
        citations=citations,
        url="https://example.com",
        abstract=abstract,
    )


class AppCoreTests(unittest.TestCase):
    def test_extract_search_query_removes_question_filler(self):
        query = app._extract_search_query(
            "What are the main approaches to early cancer detection using MRI?"
        )

        self.assertEqual(query, "approaches early cancer detection mri")

    def test_rank_results_prefers_relevance_before_year(self):
        older_relevant = paper(
            "Cancer detection with MRI",
            year="2020",
            abstract="MRI cancer detection screening model",
            citations="50",
        )
        newer_irrelevant = paper(
            "Unrelated particle physics survey",
            year="2025",
            abstract="Collider measurements",
            citations="500",
        )

        ranked = app._rank_results(
            [newer_irrelevant, older_relevant],
            "cancer detection MRI",
        )

        self.assertEqual(ranked[0], older_relevant)

    def test_context_builder_respects_budget(self):
        long_abstract = "cancer detection " * 1000
        papers = [paper(f"Paper {index}", abstract=long_abstract) for index in range(20)]

        context = app._build_synthesis_context(papers)

        self.assertLessEqual(len(context), app.SYNTHESIS_CONTEXT_CHAR_LIMIT)
        self.assertLessEqual(
            app._rough_token_count(context),
            app.SYNTHESIS_CONTEXT_TOKEN_LIMIT,
        )

    def test_clear_search_returns_all_reset_outputs(self):
        result = app.clear_search()

        self.assertEqual(len(result), 22)
        self.assertEqual(result[0], "Enter a research topic to begin.")
        self.assertIsNone(result[9])
        self.assertEqual(result[10], "")
        self.assertEqual(result[11], "")
        self.assertEqual(result[12], "")
        self.assertEqual(result[14], app.DEFAULT_ASK_ANSWER)
        self.assertEqual(result[16], app.DEFAULT_LOAD_STATUS)
        self.assertIsNone(result[18])
        self.assertEqual(result[21], app.DEFAULT_PAPER_CHAT_ANSWER)

    def test_pagination_updates_disable_edges(self):
        papers = [paper(f"Paper {index}") for index in range(app.RESULTS_PER_PAGE + 1)]

        first_prev, first_next = app._pagination_updates(papers, 0)
        second_prev, second_next = app._pagination_updates(papers, 1)

        self.assertFalse(first_prev["interactive"])
        self.assertTrue(first_next["interactive"])
        self.assertTrue(second_prev["interactive"])
        self.assertFalse(second_next["interactive"])

    def test_reconstruct_abstract_orders_openalex_index(self):
        abstract = app._reconstruct_abstract({"world": [1], "hello": [0]})

        self.assertEqual(abstract, "hello world")

    def test_normalize_doi_strips_doi_url(self):
        self.assertEqual(
            app._normalize_doi("https://doi.org/10.1234/ABC"),
            "10.1234/abc",
        )

    def test_dedupe_prefers_duplicate_with_abstract(self):
        weak = paper("Deep Learning for Cancer Detection", abstract="")
        strong = paper(
            "Deep Learning for Cancer Detection!",
            abstract="Detailed abstract",
            citations="3",
        )

        deduped = app._dedupe_results([weak, strong])

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].abstract, "Detailed abstract")

    def test_source_specific_queries(self):
        self.assertIn("ti:cancer", app._arxiv_search_query("cancer detection with MRI"))
        self.assertIn(
            "cancer[Title/Abstract]",
            app._pubmed_search_query("cancer detection with MRI"),
        )

    def test_load_selected_paper_returns_context(self):
        item = paper("Useful Paper", abstract="A clear abstract about useful results.")

        paper_text, status, summary, tab_update = app.load_selected_paper(0, [item])

        self.assertIn("Useful Paper", paper_text)
        self.assertIn("Loaded", status)
        self.assertEqual(summary, "")
        self.assertEqual(tab_update["selected"], "summarize")

    def test_export_results_csv_creates_file(self):
        path = app.export_results_csv([paper("Exportable Paper")])

        self.assertIsNotNone(path)
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("Exportable Paper", content)

    def test_combine_paper_context_includes_results_section(self):
        context = app._combine_paper_context("Abstract text", "Result text")

        self.assertIn("Abstract text", context)
        self.assertIn("Results / Findings", context)
        self.assertIn("Result text", context)

    def test_export_summary_markdown_includes_results(self):
        path = app.export_summary_markdown("Abstract text", "Result text", "Summary text")

        self.assertIsNotNone(path)
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("Results / Findings", content)
        self.assertIn("Result text", content)

    def test_modal_request_error_uses_response_detail(self):
        response = Mock()
        response.json.return_value = {"detail": "Bad input"}
        exc = app.requests.HTTPError(response=response)

        self.assertEqual(
            app._modal_request_error_message(exc, "Modal"),
            "Modal: Bad input",
        )


if __name__ == "__main__":
    unittest.main()
