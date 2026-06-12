import unittest

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

        self.assertEqual(len(result), 16)
        self.assertEqual(result[0], "Enter a research topic to begin.")
        self.assertEqual(result[10], "")
        self.assertEqual(result[11], "")
        self.assertEqual(result[12], app.DEFAULT_ASK_ANSWER)
        self.assertEqual(result[14], app.DEFAULT_LOAD_STATUS)

    def test_pagination_updates_disable_edges(self):
        papers = [paper(f"Paper {index}") for index in range(app.RESULTS_PER_PAGE + 1)]

        first_prev, first_next = app._pagination_updates(papers, 0)
        second_prev, second_next = app._pagination_updates(papers, 1)

        self.assertFalse(first_prev["interactive"])
        self.assertTrue(first_next["interactive"])
        self.assertTrue(second_prev["interactive"])
        self.assertFalse(second_next["interactive"])


if __name__ == "__main__":
    unittest.main()
