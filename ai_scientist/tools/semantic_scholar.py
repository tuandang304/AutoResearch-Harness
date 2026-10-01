import os
import requests
import time
import warnings
from typing import Dict, List, Optional, Union

import backoff

from ai_scientist.tools.base_tool import BaseTool

# Bound every Semantic Scholar call: a per-request timeout plus a finite retry
# budget, so a stalled connection or persistent 429s cannot hang a run forever.
S2_REQUEST_TIMEOUT = 30  # seconds per HTTP request
S2_MAX_TRIES = 8
S2_MAX_TIME = 300  # seconds across all retries of one search
S2_RETRY_EXCEPTIONS = (
    requests.exceptions.HTTPError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)
OPENALEX_MAX_TRIES = 2  # then Crossref is searched instead


def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class SemanticScholarSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchSemanticScholar",
        description: str = (
            "Search for relevant literature using Semantic Scholar. "
            "Provide a search query to find relevant papers."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.S2_API_KEY = os.getenv("S2_API_KEY")
        if not self.S2_API_KEY:
            warnings.warn(
                "No Semantic Scholar API key found. Requests will be subject to stricter rate limits. "
                "Set the S2_API_KEY environment variable for higher limits."
            )

    def use_tool(self, query: str) -> Optional[str]:
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    @backoff.on_exception(
        backoff.expo,
        S2_RETRY_EXCEPTIONS,
        max_tries=S2_MAX_TRIES,
        max_time=S2_MAX_TIME,
        on_backoff=on_backoff,
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        if not query:
            return None
        
        headers = {}
        if self.S2_API_KEY:
            headers["X-API-KEY"] = self.S2_API_KEY
        
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={
                "query": query,
                "limit": self.max_results,
                "fields": "title,authors,venue,year,abstract,citationCount",
            },
            timeout=S2_REQUEST_TIMEOUT,
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(f"Response Content: {rsp.text[:500]}")
        rsp.raise_for_status()
        results = rsp.json()
        total = results.get("total", 0)
        if total == 0:
            return None

        papers = results.get("data", [])
        # Sort papers by citationCount in descending order
        papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
        return papers

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = ", ".join(
                [author.get("name", "Unknown") for author in paper.get("authors", [])]
            )
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {authors}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Number of citations: {paper.get("citationCount", "N/A")}
Abstract: {paper.get("abstract", "No abstract available.")}"""
            )
        return "\n\n".join(paper_strings)


def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    """Literature search for citations. Semantic Scholar with S2_API_KEY; otherwise
    OpenAlex (no key needed), because Semantic Scholar's shared anonymous pool
    answered almost every request with HTTP 429 in practice. Crossref is the
    fallback when OpenAlex search is unavailable."""
    if not query:
        return None
    if os.getenv("S2_API_KEY"):
        return _search_semantic_scholar(query, result_limit)
    try:
        return search_openalex(query, result_limit)
    except S2_RETRY_EXCEPTIONS as exc:
        # OpenAlex search answers 503 "temporarily unavailable" for long periods.
        print(f"OpenAlex unavailable ({exc}); searching Crossref instead")
        return search_crossref(query, result_limit)


def _bibtex_escape(text):
    return str(text).replace("\\", " ").replace("{", "").replace("}", "")


def _paper(title, authors, venue, year, abstract, citations, conference, doi):
    """A search hit in the Semantic Scholar shape used by the writers."""
    import re
    import unicodedata

    def ascii_word(text):
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        return re.sub(r"[^A-Za-z0-9]", "", text).lower()

    last = ascii_word(authors[0].split()[-1]) if authors else "anon"
    first = next((ascii_word(w) for w in title.split() if len(ascii_word(w)) > 3), "paper")
    key = f"{last}{year or ''}{first}"
    entry = "inproceedings" if conference else "article"
    fields = {"title": title, "author": " and ".join(authors), "year": year,
              ("booktitle" if entry == "inproceedings" else "journal"): venue}
    if doi:
        fields["doi"] = doi
    body = ",\n".join(f"  {k} = {{{_bibtex_escape(v)}}}" for k, v in fields.items() if v)
    return {"title": title, "authors": [{"name": a} for a in authors], "venue": venue,
            "year": year, "abstract": abstract, "citationCount": citations,
            "citationStyles": {"bibtex": f"@{entry}{{{key},\n{body}\n}}"}}


def _openalex_paper(work):
    """An OpenAlex work in the Semantic Scholar shape used by the writers."""
    authors = [a.get("author", {}).get("display_name") for a in work.get("authorships") or []]
    source = (work.get("primary_location") or {}).get("source") or {}
    words = work.get("abstract_inverted_index") or {}
    positions = sorted((i, w) for w, idx in words.items() for i in idx)
    return _paper(
        work.get("display_name") or work.get("title") or "", [a for a in authors if a],
        source.get("display_name") or "", work.get("publication_year"),
        " ".join(w for _, w in positions) or None, work.get("cited_by_count"),
        source.get("type") == "conference",
        (work.get("doi") or "").replace("https://doi.org/", ""),
    )


def _crossref_paper(item):
    """A Crossref work in the Semantic Scholar shape used by the writers."""
    import re

    authors = [" ".join(p for p in (a.get("given"), a.get("family")) if p) or a.get("name")
               for a in item.get("author") or []]
    year = ((item.get("issued") or {}).get("date-parts") or [[None]])[0][0]
    abstract = re.sub(r"<[^>]+>", "", item.get("abstract") or "").strip() or None
    return _paper(
        (item.get("title") or [""])[0], [a for a in authors if a],
        (item.get("container-title") or [""])[0], year, abstract,
        item.get("is-referenced-by-count"), item.get("type") == "proceedings-article",
        item.get("DOI") or "",
    )


@backoff.on_exception(
    backoff.expo,
    S2_RETRY_EXCEPTIONS,
    max_tries=OPENALEX_MAX_TRIES,
    on_backoff=on_backoff,
)
def search_openalex(query, result_limit=10) -> Union[None, List[Dict]]:
    params = {"search": query, "per-page": result_limit}
    if os.getenv("OPENALEX_MAILTO"):  # optional "polite pool" contact address
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    rsp = requests.get("https://api.openalex.org/works", params=params,
                       timeout=S2_REQUEST_TIMEOUT)
    print(f"OpenAlex response status: {rsp.status_code}")
    rsp.raise_for_status()
    papers = [_openalex_paper(w) for w in rsp.json().get("results") or []
              if w.get("display_name")]
    time.sleep(0.2)
    return papers or None


@backoff.on_exception(
    backoff.expo,
    S2_RETRY_EXCEPTIONS,
    max_tries=S2_MAX_TRIES,
    max_time=S2_MAX_TIME,
    on_backoff=on_backoff,
)
def search_crossref(query, result_limit=10) -> Union[None, List[Dict]]:
    params = {"query.bibliographic": query, "rows": result_limit,
              "select": "DOI,title,author,container-title,issued,type,abstract,"
                        "is-referenced-by-count"}
    if os.getenv("OPENALEX_MAILTO"):  # the same optional contact address
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    rsp = requests.get("https://api.crossref.org/works", params=params,
                       timeout=S2_REQUEST_TIMEOUT)
    print(f"Crossref response status: {rsp.status_code}")
    rsp.raise_for_status()
    papers = [_crossref_paper(w) for w in rsp.json().get("message", {}).get("items") or []
              if w.get("title")]
    time.sleep(0.2)
    return papers or None


@backoff.on_exception(
    backoff.expo,
    S2_RETRY_EXCEPTIONS,
    max_tries=S2_MAX_TRIES,
    max_time=S2_MAX_TIME,
    on_backoff=on_backoff,
)
def _search_semantic_scholar(query, result_limit=10) -> Union[None, List[Dict]]:
    S2_API_KEY = os.getenv("S2_API_KEY")
    headers = {}
    if not S2_API_KEY:
        warnings.warn(
            "No Semantic Scholar API key found. Requests will be subject to stricter rate limits."
        )
    else:
        headers["X-API-KEY"] = S2_API_KEY
    
    if not query:
        return None
    
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers=headers,
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
        timeout=S2_REQUEST_TIMEOUT,
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers
