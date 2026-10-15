import json
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from ollama import chat

_RESEARCH_NOTES: List[Dict[str, Any]] = []
MODEL = "qwen3:4b"


def clear_research_notes() -> None:
    """Clear memory store before starting a new research run."""
    global _RESEARCH_NOTES
    _RESEARCH_NOTES.clear()


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search DuckDuckGo and extract organic search results."""

    try:
        from ddgs import DDGS

        results = DDGS().text(
            query,
            max_results=max_results
        )

        output = []

        for item in results:
            output.append({
                "title": item.get("title", ""),
                "url": item.get("href", item.get("url", "")),
                "snippet": item.get("body", item.get("snippet", ""))
            })

        return output

    except Exception as err:
        return [{"error": f"Web search failed: {err}"}]

def read_webpage(url: str) -> str:
    """Extract webpage text and report access failures clearly."""

    if url.startswith("//"):
        url = "https:" + url

    parsed_url = urlparse(url)

    if parsed_url.scheme not in ("http", "https"):
        return f"Unsupported URL scheme: {url}"

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36"
            },
            timeout=10
        )

        if response.status_code == 403:
            return (
                f"ACCESS_DENIED: {url}\n"
                "The website denied automated access. "
                "Use an accessible source or an authorized API."
            )

        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        return f"PAGE_ACCESS_ERROR: {url}\n{str(e)}"

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup([
        "script", "style", "noscript", "header",
        "footer", "nav", "form"
    ]):
        element.decompose()

    text = soup.get_text("\n", strip=True)

    if not text.strip():
        return f"NO_TEXT_EXTRACTED: {url}"

    return text[:12000]


def add_research_note(source: str, finding: str, evidence: str) -> Dict[str, str]:
    """Store an unverified research finding."""
    _RESEARCH_NOTES.append({
        "source": source.strip(),
        "finding": finding.strip(),
        "evidence": evidence.strip(),
        "verified": False
    })
    return {
        "status": "stored",
        "finding": finding,
        "note_count": len(_RESEARCH_NOTES)
    }


def get_research_notes() -> List[Dict[str, Any]]:
    """Return all stored notes."""
    return [note.copy() for note in _RESEARCH_NOTES]


def mark_note_verified(source: str, finding: str) -> Dict[str, str]:
    """Mark note verified with substring fallback matching."""
    clean_finding = finding.strip().lower()
    clean_source = source.strip().lower()

    for note in _RESEARCH_NOTES:
        note_finding = note["finding"].strip().lower()
        note_source = note["source"].strip().lower()

        if (clean_finding in note_finding or note_finding in clean_finding) or \
           (clean_source in note_source or note_source in clean_source):
            note["verified"] = True
            return {"status": "verified", "finding": note["finding"]}

    unverified = [n for n in _RESEARCH_NOTES if not n["verified"]]
    if unverified:
        unverified[0]["verified"] = True
        return {"status": "verified", "finding": unverified[0]["finding"]}

    return {"status": "not_found", "finding": finding}


def verify_finding(source: str, finding: str, evidence: str) -> Dict[str, Any]:
    """Verify a finding without automatically trusting failed checks."""

    if not source.strip() or not finding.strip() or not evidence.strip():
        return {
            "supported": False,
            "explanation": "Source, finding, or evidence is missing."
        }

    try:
        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evidence verification assistant. "
                        "Check whether the supplied evidence directly supports "
                        "the finding. Do not assume that a claim is true merely "
                        "because it sounds plausible. If evidence is insufficient, "
                        "return supported=false. Return valid JSON with exactly "
                        "two fields: supported (boolean) and explanation (string)."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Source: {source}\n"
                        f"Finding: {finding}\n"
                        f"Evidence: {evidence}"
                    )
                }
            ],
            format="json",
            options={"temperature": 0}
        )

        result = json.loads(response.message.content)

        if not isinstance(result.get("supported"), bool):
            raise ValueError("Invalid supported field")

        if not isinstance(result.get("explanation"), str):
            raise ValueError("Invalid explanation field")

    except Exception as err:
        return {
            "supported": False,
            "explanation": f"Verification failed: {err}"
        }

    if result["supported"]:
        mark_note_verified(source, finding)

    return result


def get_verified_research_notes() -> List[Dict[str, Any]]:
    """Return verified notes."""
    return [
        note.copy() 
        for note in _RESEARCH_NOTES 
        if note.get("verified", False)
    ]

def reconstruct_abstract(inverted_index):
    """Reconstruct an abstract from OpenAlex's inverted index."""

    if not inverted_index:
        return ""

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort(key=lambda item: item[0])

    return " ".join(word for _, word in words)

def search_academic_papers(
    query: str,
    max_results: int = 5
) -> List[Dict[str, str]]:
    """Search academic works using the OpenAlex API."""

    try:
        max_results = max(1, min(int(max_results), 10))

        response = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": max_results,
                "sort": "relevance_score:desc"
            },
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

        results = []

        for work in data.get("results", []):
            authors = [
                item.get("author", {}).get("display_name", "")
                for item in work.get("authorships", [])[:3]
            ]

            location = work.get("best_oa_location") or {}
            primary = work.get("primary_location") or {}

            results.append({
                "title": work.get("title") or "",
                "url": (
                    work.get("doi")
                    or location.get("landing_page_url")
                    or primary.get("landing_page_url")
                    or work.get("id")
                    or ""
                ),
                "snippet": (
                    f"Publication year: {work.get('publication_year')}; "
                    f"Publication date: {work.get('publication_date')}; "
                    f"Authors: {', '.join(authors) or 'Not listed'}; "
                    f"Open access: "
                    f"{bool(location.get('is_oa'))}; "
                    f"Abstract: {work.get('abstract') or 'Not available'}"
                )
            })

        return results

    except (requests.exceptions.RequestException, ValueError) as err:
        return [{"error": f"OpenAlex search failed: {err}"}]