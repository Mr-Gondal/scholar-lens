import unittest
from unittest.mock import Mock, patch
from pathlib import Path

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

    def test_split_search_queries_accepts_multiple_keywords(self):
        queries = app._split_search_queries("aerosols, cloud feedback\nsatellite rainfall; aerosols")

        self.assertEqual(queries, ["aerosols", "cloud feedback", "satellite rainfall"])

    def test_collect_results_merges_multiple_keyword_searches(self):
        first = paper("Aerosol cloud interactions", abstract="aerosol cloud")
        second = paper("Satellite rainfall retrieval", abstract="satellite rainfall")

        with patch(
            "app._collect_single_query_results",
            side_effect=[([first], []), ([second], [])],
        ) as mocked:
            results, warnings = app._collect_results("aerosols, satellite rainfall")

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual({result.title for result in results}, {first.title, second.title})
        self.assertEqual(warnings, [])

    def test_search_all_sources_reports_multiple_keyword_searches(self):
        with patch(
            "app._collect_results",
            return_value=([paper("Aerosol cloud interactions")], []),
        ):
            status, *_ = app.search_all_sources("aerosols, cloud feedback")

        self.assertIn("2", status)
        self.assertIn("keyword searches", status)

    def test_render_result_insights_handles_results(self):
        panel = app._render_result_insights([paper("Aerosol cloud interactions")])

        self.assertIn("Papers", panel)
        self.assertIn("Top Source", panel)

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

        self.assertEqual(len(result), 28)
        self.assertEqual(result[0], "Enter a research topic to begin.")
        self.assertIsNone(result[9])
        self.assertEqual(result[10], "")
        self.assertEqual(result[11], "")
        self.assertEqual(result[12], "")
        self.assertEqual(result[14], app.DEFAULT_ASK_ANSWER)
        self.assertEqual(result[16], app.DEFAULT_LOAD_STATUS)
        self.assertIsNone(result[18])
        self.assertEqual(result[21], app.DEFAULT_PAPER_CHAT_ANSWER)
        self.assertEqual(result[25], app.DEFAULT_COMPARE_ANSWER)
        self.assertIn("Literature Constellation", result[26])

    def test_pagination_updates_disable_edges(self):
        papers = [paper(f"Paper {index}") for index in range(app.RESULTS_PER_PAGE + 1)]

        first_prev, first_next = app._pagination_updates(papers, 0)
        second_prev, second_next = app._pagination_updates(papers, 1)

        self.assertFalse(first_prev["interactive"])
        self.assertTrue(first_next["interactive"])
        self.assertTrue(second_prev["interactive"])
        self.assertFalse(second_next["interactive"])

    def test_compare_selector_updates_use_display_choices(self):
        papers = [paper("Paper A"), paper("Paper B")]

        left, right = app._compare_selector_updates(papers)

        self.assertEqual(left["value"], "1. Paper A")
        self.assertEqual(right["value"], "2. Paper B")

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

    def test_rubric_proof_panel_mentions_judging_evidence(self):
        panel = app._render_rubric_proof()

        self.assertIn("Backyard AI proof points", panel)
        self.assertIn("Real professor workflow", panel)
        self.assertIn("NVIDIA Nemotron fit", panel)
        self.assertIn(app.MODEL_DISPLAY_NAME, panel)

    def test_public_model_story_is_nvidia_nemotron(self):
        self.assertEqual(
            app.MODEL_ID,
            "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        )
        self.assertEqual(app.MODEL_PROVIDER_BADGE, "Powered by NVIDIA Nemotron on Modal")
        self.assertIn("Nemotron", app.MODEL_DISPLAY_NAME)

    def test_modal_default_model_is_nemotron_nano(self):
        modal_source = Path(app.__file__).with_name("modal_inference.py").read_text(
            encoding="utf-8",
        )

        self.assertIn(
            '"nvidia/Llama-3.1-Nemotron-Nano-8B-v1"',
            modal_source,
        )
        self.assertIn("trust_remote_code=True", modal_source)
        self.assertIn("enforce_eager=True", modal_source)
        self.assertIn("Use only the supplied context", modal_source)

    def test_load_selected_paper_returns_context(self):
        item = paper("Useful Paper", abstract="A clear abstract about useful results.")

        paper_text, status, summary, tab_update = app.load_selected_paper(0, [item])

        self.assertIn("Useful Paper", paper_text)
        self.assertIn("Loaded", status)
        self.assertEqual(summary, "")
        self.assertEqual(tab_update["selected"], "summarize")

    def test_summarize_now_loads_tab_without_modal_call(self):
        item = paper("Useful Paper", abstract="A clear abstract about useful results.")
        with patch("app.summarize_with_modal") as mocked:
            paper_text, status, summary, tab_update, *_ = app.load_selected_paper_reset_chat(
                0,
                [item],
            )

        mocked.assert_not_called()
        self.assertIn("Useful Paper", paper_text)
        self.assertIn("Loaded", status)
        self.assertIn("Click Summarize with AI", summary)
        self.assertEqual(tab_update["selected"], "summarize")

    def test_row_selection_loads_without_modal_call(self):
        item = paper("Useful Paper", abstract="A clear abstract about useful results.")
        with patch("app.summarize_with_modal") as mocked:
            paper_text, status, summary, tab_update, *_ = app.summarize_row_selection(
                "0",
                [item],
            )

        mocked.assert_not_called()
        self.assertIn("Useful Paper", paper_text)
        self.assertIn("Loaded", status)
        self.assertIn("Click Summarize with AI", summary)
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

    def test_get_first_author_accepts_string_or_list(self):
        self.assertEqual(app.get_first_author("Ada Lovelace, Alan Turing"), "Ada Lovelace")
        self.assertEqual(app.get_first_author(["Grace Hopper", "Katherine Johnson"]), "Grace Hopper")

    def test_search_result_constellation_marks_keyword_fallback(self):
        graph = app.build_constellation_from_papers(
            "connectome",
            [
                paper("Functional connectome graph theory", abstract="modularity graph network"),
                paper("Resting state connectome modularity", abstract="resting fmri modularity"),
            ],
        )

        self.assertTrue(graph["data_completeness"]["keyword_fallback_used"])
        self.assertEqual(graph["data_completeness"]["paper_count"], 2)
        self.assertIn("nodes", graph)

    def test_constellation_community_ids_match_nodes(self):
        graph = app.build_constellation_from_papers(
            "mixed methods",
            [
                paper("Functional graph modularity", abstract="graph modularity community"),
                paper("Clinical disease cohort", abstract="clinical disease disorder"),
                paper("Diffusion tractography", abstract="structural diffusion tractography"),
            ],
        )
        community_ids = {community["id"] for community in graph["communities"]}

        self.assertTrue({node["community"] for node in graph["nodes"]}.issubset(community_ids))

    def test_constellation_render_has_nonblank_fallback(self):
        graph = app.build_constellation_from_papers(
            "connectome",
            [paper("Functional connectome graph theory", abstract="modularity graph network")],
        )
        html = app._render_constellation_html(graph)

        self.assertIn("Literature Constellation", html)
        self.assertIn("CONNECTED LITERATURE MAP", html)
        self.assertIn("canvas", html)
        self.assertIn("function showDetail", html)
        self.assertIn("if (node) showDetail(node)", html)
        self.assertNotIn("const labeled", html)
        self.assertNotIn("fillText(label", html)
        self.assertNotIn("Connectome Constellation", html)

    def test_compare_prompt_names_nemotron(self):
        results = [
            paper("Paper A", abstract="A studies rainfall with satellite data."),
            paper("Paper B", abstract="B studies rainfall with station data."),
        ]
        with patch("app.synthesize_with_modal", return_value="comparison") as mocked:
            result = app.compare_papers_with_ai(0, 1, results)

        self.assertEqual(result, "comparison")
        prompt = mocked.call_args.args[0]
        self.assertIn(app.MODEL_DISPLAY_NAME, prompt)
        self.assertIn("Use only the provided metadata", prompt)

    def test_export_corpus_zip_includes_graph_json(self):
        graph = app.build_constellation_from_papers(
            "connectome",
            [paper("Functional connectome graph theory", abstract="modularity graph network")],
        )

        path = app.export_corpus_zip(graph)

        self.assertIsNotNone(path)
        with app.zipfile.ZipFile(path) as archive:
            self.assertIn("graph.json", archive.namelist())
            self.assertIn("data-completeness.json", archive.namelist())


if __name__ == "__main__":
    unittest.main()
