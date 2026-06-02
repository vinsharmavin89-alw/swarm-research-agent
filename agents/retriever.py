from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel
from ddgs import DDGS
import json

load_dotenv()

client = Groq()


class SearchResult(BaseModel):
    keyword: str
    summary: str
    key_facts: list[str]


class RetrieverOutput(BaseModel):
    query: str
    results: list[SearchResult]
    total_keywords_searched: int


def search_web(keyword: str) -> str:
    """
    Uses DuckDuckGo search library — much more reliable
    than scraping HTML. Returns real search snippets.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(keyword, max_results=5))

        if not results:
            return f"No results found for: {keyword}"

        snippets = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            if body:
                snippets.append(f"{title}: {body}")

        return " | ".join(snippets)

    except Exception as e:
        return f"Search error for '{keyword}': {str(e)}"


def summarize_with_llm(keyword: str, raw_text: str) -> SearchResult:
    """
    Uses Groq/LLaMA to extract clean summary and key facts
    from raw search results.
    """

    system_prompt = """You are a Retriever Agent in a multi-agent research system.

You receive a search keyword and raw search result text.
Extract useful information and return ONLY a raw JSON object.
No markdown, no code fences, no explanation. Just JSON starting with { and ending with }

Return exactly this structure:
{
  "keyword": "the search keyword exactly as given",
  "summary": "2-3 sentence summary of what was found",
  "key_facts": ["specific fact 1", "specific fact 2", "specific fact 3"]
}

Rules:
- summary: 2-3 sentences, plain English, based only on the text provided
- key_facts: 2 to 5 specific facts, numbers, names, or data points from the text
- If truly nothing useful, say so honestly but still return valid JSON with empty key_facts list
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Search keyword: {keyword}\n\nRaw search results:\n{raw_text[:3000]}\n\nExtract key information and return JSON."
            }
        ]
    )

    raw_json = response.choices[0].message.content.strip()

    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()

    data = json.loads(raw_json)
    return SearchResult(**data)


def run_retriever(original_query: str, search_keywords: list[str]) -> RetrieverOutput:
    """
    Retriever Agent — searches web for each keyword
    and summarizes findings using LLM.
    """

    print(f"\n  Searching {len(search_keywords)} keywords...")
    results = []

    for i, keyword in enumerate(search_keywords, 1):
        print(f"  [{i}/{len(search_keywords)}] Searching: '{keyword}'")

        raw_text = search_web(keyword)
        print(f"           Raw text: {len(raw_text)} chars")

        result = summarize_with_llm(keyword, raw_text)
        results.append(result)
        print(f"           Facts found: {len(result.key_facts)}")

    return RetrieverOutput(
        query=original_query,
        results=results,
        total_keywords_searched=len(search_keywords)
    )


if __name__ == "__main__":
    test_query = "What are the latest trends in electric vehicles in India in 2025?"
    test_keywords = [
        "Electric vehicle trends India 2025",
        "Indian electric vehicle market forecast",
        "Government initiatives for electric vehicles in India"
    ]

    print("=" * 55)
    print("  RETRIEVER AGENT — SwarmIQ")
    print("=" * 55)

    result = run_retriever(test_query, test_keywords)

    print(f"\nResults:\n")
    for r in result.results:
        print(f"  Keyword : {r.keyword}")
        print(f"  Summary : {r.summary}")
        print(f"  Facts   :")
        for f in r.key_facts:
            print(f"    - {f}")
        print()