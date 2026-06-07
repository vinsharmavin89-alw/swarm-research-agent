from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel
import json

load_dotenv()

client = Groq()


class FactCheck(BaseModel):
    fact: str
    verdict: str        # "confirmed", "uncertain", or "contradicted"
    confidence: int     # 0-100
    reason: str


class ValidatorOutput(BaseModel):
    query: str
    overall_reliability: str   # "high", "medium", or "low"
    confidence_score: int      # 0-100 overall
    fact_checks: list[FactCheck]
    key_insights: list[str]
    gaps_identified: list[str]
    recommendation: str


def run_validator(query: str, retriever_results: list) -> ValidatorOutput:
    """
    Validator Agent — the quality control layer of the swarm.

    What it does:
      1. Takes all facts gathered by the Retriever
      2. Cross-checks facts against each other for consistency
      3. Scores confidence on each fact
      4. Identifies gaps and contradictions
      5. Gives an overall reliability score

    This is what separates SwarmIQ from a simple chatbot.
    Judges will immediately notice this layer exists.

    Input:  query + list of SearchResult objects from Retriever
    Output: ValidatorOutput with fact checks and confidence scores
    """

    # Build a structured summary of everything the Retriever found
    research_summary = f"Research Query: {query}\n\n"
    all_facts = []

    for i, result in enumerate(retriever_results, 1):
        research_summary += f"Source {i} — Keyword: '{result.keyword}'\n"
        research_summary += f"Summary: {result.summary}\n"
        research_summary += "Facts found:\n"
        for fact in result.key_facts:
            research_summary += f"  - {fact}\n"
            all_facts.append(fact)
        research_summary += "\n"

    system_prompt = """You are a Validator Agent in a multi-agent research system.

You receive research findings gathered by a Retriever Agent.
Your job is to validate the quality, consistency, and reliability of the information.

You must respond with ONLY a raw JSON object.
No markdown, no code fences, no explanation. Just JSON starting with { and ending with }

Return exactly this structure:
{
  "query": "the original research query",
  "overall_reliability": "high",
  "confidence_score": 75,
  "fact_checks": [
    {
      "fact": "the fact being checked",
      "verdict": "confirmed",
      "confidence": 85,
      "reason": "why you gave this verdict"
    }
  ],
  "key_insights": ["insight 1", "insight 2", "insight 3"],
  "gaps_identified": ["gap 1", "gap 2"],
  "recommendation": "one sentence on how reliable this research is overall"
}

Rules:
- overall_reliability: must be exactly one of: high, medium, low
- confidence_score: 0-100 integer for overall research quality
- fact_checks: check every fact provided, give verdict + confidence + reason
- verdict must be exactly one of: confirmed, uncertain, contradicted
- key_insights: 2-4 most important takeaways from ALL the research combined
- gaps_identified: 1-3 important things that were NOT found but should have been
- recommendation: one clear sentence about reliability and what to trust
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.1,  # very low — we want consistent validation
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Please validate this research:\n\n{research_summary}"
            }
        ]
    )

    raw_json = response.choices[0].message.content.strip()

    # Clean markdown fences if model adds them
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()
    import re
    raw_json = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw_json)
    data = json.loads(raw_json)
    return ValidatorOutput(**data)


if __name__ == "__main__":
    # Simulate what the Retriever would pass to us
    from pydantic import BaseModel as BM

    class MockSearchResult(BM):
        keyword: str
        summary: str
        key_facts: list[str]

    mock_results = [
        MockSearchResult(
            keyword="Electric vehicle trends India 2025",
            summary="India's EV market soared to 2.3 million units in 2025, with 8% of total vehicle sales being electric. Growth is driven by policy shifts and competitive dynamics.",
            key_facts=[
                "India's EV market soared to 2.3 million units in 2025",
                "8% of total vehicle sales in 2025 were electric vehicles",
                "Tata, Mahindra, MG, and Kia are launching new electric models",
                "Growth driven by policy shifts and evolving consumer awareness"
            ]
        ),
        MockSearchResult(
            keyword="Indian electric vehicle market forecast",
            summary="Limited data found for this specific keyword.",
            key_facts=[]
        ),
        MockSearchResult(
            keyword="Government initiatives for electric vehicles in India",
            summary="The Indian government launched PM E-DRIVE scheme and FAME-II subsidies to promote EV adoption and infrastructure.",
            key_facts=[
                "PM E-DRIVE scheme launched in 2024",
                "FAME-II subsidies available for electric vehicle buyers",
                "Import duty rationalizations and PLI incentives provided",
                "Government aims to make India a global EV leader"
            ]
        )
    ]

    test_query = "What are the latest trends in electric vehicles in India in 2025?"

    print("=" * 55)
    print("  VALIDATOR AGENT — SwarmIQ")
    print("=" * 55)
    print(f"Query: {test_query}")
    print(f"Validating {sum(len(r.key_facts) for r in mock_results)} facts across {len(mock_results)} sources...\n")

    result = run_validator(test_query, mock_results)

    print(f"Overall reliability : {result.overall_reliability.upper()}")
    print(f"Confidence score    : {result.confidence_score}/100")
    print(f"\nFact checks ({len(result.fact_checks)}):")
    for fc in result.fact_checks:
        icon = "+" if fc.verdict == "confirmed" else "?" if fc.verdict == "uncertain" else "x"
        print(f"  [{icon}] {fc.fact[:60]}...")
        print(f"       Verdict: {fc.verdict} ({fc.confidence}%) — {fc.reason}")
    print(f"\nKey insights:")
    for insight in result.key_insights:
        print(f"  * {insight}")
    print(f"\nGaps identified:")
    for gap in result.gaps_identified:
        print(f"  ! {gap}")
    print(f"\nRecommendation: {result.recommendation}")
    print("\nValidator Agent done. Ready to pass to Writer.")