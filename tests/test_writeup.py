"""Offline regression tests for the paper-writing stages (no model or network calls)."""
from contextlib import ExitStack, redirect_stdout
import io
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import Mock, create_autospec, patch

import requests

from ai_scientist import llm
from ai_scientist import perform_icbinb_writeup as icbinb
from ai_scientist import perform_writeup as normal
from ai_scientist.tools import semantic_scholar

ROOT = Path(__file__).resolve().parents[1]
HAVE_LATEX = all(shutil.which(tool) for tool in ("pdflatex", "bibtex"))


def fake_llm(responses):
    """Autospec of the real get_response_from_llm, so wrong kwargs raise TypeError."""
    mock = create_autospec(llm.get_response_from_llm)
    mock.side_effect = [(text, [{"role": "assistant", "content": text}]) for text in responses]
    return mock


def latex_reply(body):
    return f"```latex\n\\documentclass{{article}}\\begin{{document}}{body}\\end{{document}}\n```"


class WriteupTestCase(unittest.TestCase):
    def setUp(self):
        # perform_writeup copies the blank templates by repo-relative path.
        self._cwd = os.getcwd()
        os.chdir(ROOT)
        self.addCleanup(os.chdir, self._cwd)
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base, True)
        self.output = io.StringIO()

    def run_quietly(self, fn, *args, **kwargs):
        with redirect_stdout(self.output):
            return fn(*args, **kwargs)


class NormalWriteupTests(WriteupTestCase):
    def test_writeup_calls_llm_with_its_real_signature(self):
        responses = ["No more citations needed", latex_reply("draft"), "I am done"]
        request = fake_llm(responses)

        def compile_stub(cwd, pdf_file, timeout=None):
            Path(pdf_file).write_bytes(b"%PDF-1.4")
            return True

        with ExitStack() as stack:
            stack.enter_context(patch.object(normal, "get_response_from_llm", request))
            stack.enter_context(patch.object(normal, "create_client", return_value=(Mock(), "m")))
            stack.enter_context(patch.object(normal, "create_vlm_client", return_value=(Mock(), "v")))
            stack.enter_context(patch.object(normal, "compile_latex", side_effect=compile_stub))
            stack.enter_context(patch.object(normal, "detect_pages_before_impact", return_value=None))
            stack.enter_context(patch.object(normal, "run_chktex", return_value=""))
            ok = self.run_quietly(
                normal.perform_writeup, str(self.base), num_cite_rounds=1, n_writeup_reflections=1
            )
        self.assertTrue(ok)
        self.assertEqual(request.call_count, len(responses))
        for call in request.call_args_list:
            self.assertIn("prompt", call.kwargs)


class IcbinbWriteupTests(WriteupTestCase):
    def patch_pipeline(self, stack, request, compile_side_effect):
        stack.enter_context(patch.object(icbinb, "get_response_from_llm", request))
        stack.enter_context(patch.object(icbinb, "create_client", return_value=(Mock(), "m")))
        stack.enter_context(patch.object(icbinb, "create_vlm_client", return_value=(Mock(), "v")))
        stack.enter_context(patch.object(icbinb, "compile_latex", side_effect=compile_side_effect))
        stack.enter_context(patch.object(icbinb, "run_chktex", return_value=""))
        reviews = [
            stack.enter_context(patch.object(icbinb, name, return_value="review", __name__=name))
            for name in (
                "perform_imgs_cap_ref_review",
                "detect_duplicate_figures",
                "perform_imgs_cap_ref_review_selection",
            )
        ]
        return reviews

    def test_final_prompt_is_formatted_and_missing_final_latex_falls_back(self):
        request = fake_llm([
            latex_reply("draft"),
            latex_reply("revised"),
            "I am done",
            "The paper already fits.",  # final page-limit reply without a latex block
        ])

        def compile_stub(cwd, pdf_file, timeout=None):
            Path(pdf_file).write_bytes(b"%PDF-1.4 " + Path(cwd, "template.tex").read_bytes())
            return True

        with ExitStack() as stack:
            self.patch_pipeline(stack, request, compile_stub)
            stack.enter_context(
                patch.object(icbinb, "get_reflection_page_info", return_value="PAGE-INFO-SENTINEL")
            )
            ok = self.run_quietly(
                icbinb.perform_writeup, str(self.base), citations_text="", n_writeup_reflections=1
            )
        self.assertTrue(ok)
        final_prompt = request.call_args_list[-1].kwargs["prompt"]
        self.assertIn("PAGE-INFO-SENTINEL", final_prompt)
        self.assertNotIn("{reflection_page_info}", final_prompt)
        name = self.base.name
        final_pdf = self.base / f"{name}_reflection_final_page_limit.pdf"
        self.assertEqual(final_pdf.read_bytes(), (self.base / f"{name}_reflection1.pdf").read_bytes())
        self.assertIn("WARNING", self.output.getvalue())

    def test_failed_compile_skips_pdf_reviews_and_feeds_back_latex_errors(self):
        request = fake_llm([
            latex_reply("draft"),
            latex_reply("revised"),
            "I am done",
            "Nothing to change.",
        ])

        def failing_compile(cwd, pdf_file, timeout=None):
            Path(cwd, "template.log").write_text("! Undefined control sequence.\nl.7 \\badmacro\n")
            return False

        with ExitStack() as stack:
            reviews = self.patch_pipeline(stack, request, failing_compile)
            ok = self.run_quietly(
                icbinb.perform_writeup, str(self.base), citations_text="", n_writeup_reflections=1
            )
        # No PDF was ever produced, so there is nothing to fall back to...
        self.assertFalse(ok)
        # ...but the attempt ran to completion instead of aborting on a missing PDF.
        self.assertEqual(request.call_count, 4)
        for review in reviews:
            review.assert_not_called()
        reflection_prompt = request.call_args_list[1].kwargs["prompt"]
        self.assertIn("Undefined control sequence", reflection_prompt)

    def test_prompt_templates_format_cleanly(self):
        for module in (normal, icbinb):
            with self.subTest(module=module.__name__):
                system = module.writeup_system_message_template.format(page_limit=4)
                self.assertIn("4 pages", system)
                prompt = module.writeup_prompt.format(
                    idea_text="i", summaries="s", aggregator_code="a",
                    plot_list="p", latex_writeup="l", plot_descriptions="d",
                )
                self.assertNotRegex(prompt, r"(?<!\{)\{[a-z_]+\}(?!\})")

    def test_plot_reflection_counts_only_png_figures(self):
        from ai_scientist import perform_plotting as plotting

        script = ("```python\nimport os\nos.makedirs('figures', exist_ok=True)\n"
                  "open('figures/a.pdf', 'w').close()\nopen('figures/b.png', 'w').close()\n```")
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(plotting, "load_idea_text", return_value="idea"), \
                patch.object(plotting, "load_exp_summaries", return_value={}), \
                patch.object(plotting, "filter_experiment_summaries", return_value={}), \
                patch.object(plotting, "create_client", return_value=(Mock(), "m")), \
                patch.object(plotting, "get_response_from_llm",
                             side_effect=[(script, []), ("I am done", [])]) as request, \
                redirect_stdout(io.StringIO()):
            plotting.aggregate_plots(directory, model="m", n_reflections=1)
        self.assertIn(".png", plotting.build_aggregator_prompt("s", "i"))
        reflection = request.call_args_list[1].kwargs["prompt"]
        self.assertIn("produced 1 .png figure(s). 1 other file(s)", reflection)

    def test_seed_results_reach_plotting_and_writeup(self):
        node = {"analysis": "a", "metric": "m", "code": "c", "plot_code": "p",
                "exp_results_npy_files": ["s.npy"]}
        summaries = {"RESEARCH_SUMMARY": {
            "best node": node,
            "best node with different seeds": [node, node],
            "aggregated results of nodes with different seeds": node,
        }}
        plot = icbinb.filter_experiment_summaries(summaries, "plot_aggregation")
        seeds = plot["RESEARCH_SUMMARY"]["best node with different seeds"]
        self.assertEqual([s["exp_results_npy_files"] for s in seeds], [["s.npy"]] * 2)
        self.assertNotIn("plot_code", seeds[0])
        write = icbinb.filter_experiment_summaries(summaries, "writeup")["RESEARCH_SUMMARY"]
        self.assertEqual(write["best node with different seeds"][0], {"analysis": "a", "metric": "m"})
        self.assertIn("code", write["best node"])
        self.assertIn("aggregated results of nodes with different seeds", write)
        cite = icbinb.filter_experiment_summaries(summaries, "citation_gathering")
        self.assertEqual(list(cite["RESEARCH_SUMMARY"]), ["best node"])


class LatexToolTests(WriteupTestCase):
    def test_chktex_absence_warns_once(self):
        with patch.object(normal.shutil, "which", return_value=None), \
                patch.object(normal, "_chktex_missing_warned", False):
            first = self.run_quietly(normal.run_chktex, "template.tex")
            second = self.run_quietly(normal.run_chktex, "template.tex")
        self.assertEqual(first, normal.CHKTEX_MISSING_NOTE)
        self.assertEqual(second, normal.CHKTEX_MISSING_NOTE)
        self.assertEqual(self.output.getvalue().count("chktex was not found"), 1)

    def test_stale_filecontents_output_is_removed_before_compile(self):
        (self.base / "template.tex").write_text(
            "\\begin{filecontents}{references.bib}\n@misc{a}\n\\end{filecontents}\n"
        )
        (self.base / "references.bib").write_text("stale")
        (self.base / "keep.bib").write_text("keep")
        normal.remove_stale_filecontents(str(self.base))
        self.assertFalse((self.base / "references.bib").exists())
        self.assertTrue((self.base / "keep.bib").exists())

    def test_icbinb_template_uses_the_injected_bibliography(self):
        tex = (ROOT / "ai_scientist/blank_icbinb_latex/template.tex").read_text()
        self.assertIn("\\begin{filecontents}{references.bib}", tex)
        self.assertEqual(re.findall(r"\\bibliography\{([^}]+)\}", tex), ["references"])

    @unittest.skipUnless(HAVE_LATEX, "pdflatex/bibtex not installed")
    def test_injected_citations_resolve_across_recompiles(self):
        latex = self.base / "latex"
        shutil.copytree(ROOT / "ai_scientist/blank_icbinb_latex", latex)
        tex = latex / "template.tex"

        def add_citation(key):
            text = tex.read_text().replace(
                "\\end{filecontents}",
                f"@article{{{key},\n title={{T}},\n author={{Doe, Jane}},\n"
                f" journal={{J}},\n year={{2024}}\n}}\n\\end{{filecontents}}",
            )
            tex.write_text(text.replace("INTRO HERE", f"INTRO HERE \\citep{{{key}}}", 1))

        for index, key in enumerate(("fakefirst2024", "fakesecond2024")):
            add_citation(key)  # the second key checks that a stale references.bib is refreshed
            pdf = self.base / f"out{index}.pdf"
            self.assertTrue(self.run_quietly(normal.compile_latex, str(latex), str(pdf)))
            log = (latex / "template.log").read_text(errors="ignore")
            self.assertNotIn("undefined", log.lower())
            self.assertTrue(pdf.exists())


class SemanticScholarTests(unittest.TestCase):
    def failing_response(self):
        response = Mock(status_code=429, text="rate limited")
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("429")
        return response

    def test_searches_use_timeout_and_bounded_retries(self):
        tool = semantic_scholar.SemanticScholarSearchTool.__new__(
            semantic_scholar.SemanticScholarSearchTool
        )
        tool.S2_API_KEY = None
        tool.max_results = 3
        searches = {
            "tool": lambda: tool.search_for_papers("query"),
            "function": lambda: semantic_scholar.search_for_papers("query"),
        }
        # Without a key the function tries OpenAlex briefly, then Crossref.
        tries = {"tool": semantic_scholar.S2_MAX_TRIES,
                 "function": semantic_scholar.OPENALEX_MAX_TRIES + semantic_scholar.S2_MAX_TRIES}
        for name, search in searches.items():
            with self.subTest(search=name), \
                    patch.object(semantic_scholar.requests, "get",
                                 return_value=self.failing_response()) as get, \
                    patch("time.sleep"), patch("warnings.warn"), redirect_stdout(io.StringIO()):
                with self.assertRaises(requests.exceptions.HTTPError):
                    search()
                self.assertEqual(get.call_count, tries[name])
                for call in get.call_args_list:
                    self.assertEqual(call.kwargs["timeout"], semantic_scholar.S2_REQUEST_TIMEOUT)


class OpenAlexCitationTests(unittest.TestCase):
    WORK = {
        "display_name": "You Only Look Around: Low-light Object Detection",
        "publication_year": 2024, "cited_by_count": 7, "doi": "https://doi.org/10.1/x",
        "authorships": [{"author": {"display_name": "Mingbo Hong"}},
                        {"author": {"display_name": "Shen Cheng"}}],
        "primary_location": {"source": {"display_name": "NeurIPS", "type": "conference"}},
        "abstract_inverted_index": {"Low": [0], "light": [1]},
    }

    def test_openalex_work_becomes_a_citable_paper(self):
        from ai_scientist.tools.semantic_scholar import _openalex_paper

        paper = _openalex_paper(self.WORK)
        self.assertEqual((paper["year"], paper["venue"], paper["abstract"]), (2024, "NeurIPS", "Low light"))
        bib = paper["citationStyles"]["bibtex"]
        self.assertTrue(bib.startswith("@inproceedings{hong2024only,"))
        self.assertIn("author = {Mingbo Hong and Shen Cheng}", bib)
        self.assertIn("doi = {10.1/x}", bib)

    def test_without_s2_key_openalex_is_used(self):
        from ai_scientist.tools import semantic_scholar

        response = unittest.mock.Mock(status_code=200)
        response.json.return_value = {"results": [self.WORK]}
        with unittest.mock.patch.dict(os.environ, {}, clear=False), \
                unittest.mock.patch.object(semantic_scholar.requests, "get", return_value=response) as get, \
                unittest.mock.patch.object(semantic_scholar.time, "sleep"):
            os.environ.pop("S2_API_KEY", None)
            papers = semantic_scholar.search_for_papers("low light", 3)
        self.assertEqual(get.call_args.args[0], "https://api.openalex.org/works")
        self.assertEqual(papers[0]["title"], self.WORK["display_name"])

    def test_unavailable_openalex_falls_back_to_crossref(self):
        from ai_scientist.tools import semantic_scholar

        down = unittest.mock.Mock(status_code=503)
        down.raise_for_status.side_effect = requests.exceptions.HTTPError("503")
        up = unittest.mock.Mock(status_code=200)
        up.json.return_value = {"message": {"items": [{
            "title": ["2PCNet: Two-Phase Consistency Training"], "type": "proceedings-article",
            "author": [{"given": "Mikhail", "family": "Kennerley"}], "DOI": "10.1109/x",
            "container-title": ["CVPR"], "issued": {"date-parts": [[2023, 6]]},
            "abstract": "<jats:p>Day to night.</jats:p>", "is-referenced-by-count": 3,
        }]}}
        with unittest.mock.patch.dict(os.environ, {}, clear=False), \
                unittest.mock.patch.object(semantic_scholar.requests, "get",
                                           side_effect=[down, down, up]) as get, \
                unittest.mock.patch("time.sleep"), redirect_stdout(io.StringIO()):
            os.environ.pop("S2_API_KEY", None)
            papers = semantic_scholar.search_for_papers("2PCNet", 3)
        self.assertEqual(get.call_args.args[0], "https://api.crossref.org/works")
        self.assertEqual((papers[0]["year"], papers[0]["abstract"]), (2023, "Day to night."))
        self.assertTrue(papers[0]["citationStyles"]["bibtex"].startswith(
            "@inproceedings{kennerley20232pcnet,"))


if __name__ == "__main__":
    unittest.main()
