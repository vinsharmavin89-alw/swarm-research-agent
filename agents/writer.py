from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel
import json
from datetime import datetime

load_dotenv()

client = Groq()


class WriterOutput(BaseModel):
    title: str
    executive_summary: str
    sections: list[dict]
    confidence_score: int
    reliability: str
    gaps: list[str]
    generated_at: str


def run_writer(
    query: str,
    retriever_results: list,
    validator_output
) -> WriterOutput:
    """
    Writer Agent — the final agent in the swarm.

    Takes validated research and produces a clean,
    structured report that a human can actually use.

    Input:  query + retriever results + validator output
    Output: WriterOutput with title, sections, summary
    """

    # Build complete research context for the Writer
    context = f"Original Query: {query}\n\n"
    context += f"Overall Reliability: {validator_output.overall_reliability}\n"
    context += f"Confidence Score: {validator_output.confidence_score}/100\n\n"

    context += "VALIDATED FACTS:\n"
    for fc in validator_output.fact_checks:
        if fc.verdict in ["confirmed", "uncertain"]:
            context += f"  [{fc.verdict.upper()}] {fc.fact}\n"

    context += "\nKEY INSIGHTS FROM VALIDATION:\n"
    for insight in validator_output.key_insights:
        context += f"  - {insight}\n"

    context += "\nRESEARCH GAPS:\n"
    for gap in validator_output.gaps_identified:
        context += f"  - {gap}\n"

    context += "\nDETAILED FINDINGS BY KEYWORD:\n"
    for r in retriever_results:
        if r.key_facts:
            context += f"\n  Topic: {r.keyword}\n"
            context += f"  Summary: {r.summary}\n"

    system_prompt = """You are a Writer Agent in a multi-agent research system.

You receive validated research findings and must produce a structured report.

You must respond with ONLY a raw JSON object.
No markdown, no code fences, no explanation. Just JSON starting with { and ending with }

Return exactly this structure:
{
  "title": "clear descriptive title for this research report",
  "executive_summary": "3-4 sentence overview of the most important findings",
  "sections": [
    {
      "heading": "section heading",
      "content": "2-3 paragraphs of well-written content for this section"
    }
  ],
  "confidence_score": 80,
  "reliability": "high",
  "gaps": ["gap 1", "gap 2"],
  "generated_at": "placeholder"
}

Rules:
- title: specific and descriptive, not generic
- executive_summary: the 3-4 most important things a reader needs to know
- sections: create 3-4 logical sections that organize the research findings
- Each section heading should be meaningful (e.g. "Market Growth", "Government Policy")
- Each section content should be 2-3 paragraphs of clear, professional writing
- confidence_score: copy from validator (integer 0-100)
- reliability: copy from validator (high/medium/low)
- gaps: copy the gaps from validator
- generated_at: use the string "now"
- Write for a professional audience — clear, factual, no fluff
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # bigger model for final report quality
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Write a research report based on this validated research:\n\n{context}"
            }
        ]
    )

    raw_json = response.choices[0].message.content.strip()

    # Clean markdown fences
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()

    data = json.loads(raw_json)

    # Replace placeholder timestamp with real one
    data["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return WriterOutput(**data)


def print_report(report: WriterOutput):
    """Pretty prints the final report to the terminal."""

    print("\n" + "=" * 60)
    print(f"  {report.title}")
    print("=" * 60)
    print(f"  Generated : {report.generated_at}")
    print(f"  Reliability: {report.reliability.upper()}  |  Confidence: {report.confidence_score}/100")
    print("=" * 60)

    print(f"\nEXECUTIVE SUMMARY")
    print("-" * 40)
    print(report.executive_summary)

    for section in report.sections:
        print(f"\n{section['heading'].upper()}")
        print("-" * 40)
        print(section['content'])

    if report.gaps:
        print(f"\nRESEARCH GAPS")
        print("-" * 40)
        for gap in report.gaps:
            print(f"  * {gap}")

    print("\n" + "=" * 60)
    print("  SwarmIQ Report Complete")
    print("=" * 60)


if __name__ == "__main__":
    # Simulate full pipeline with mock data from previous agents
    from pydantic import BaseModel as BM

    class MockSearchResult(BM):
        keyword: str
        summary: str
        key_facts: list[str]

    class MockFactCheck(BM):
        fact: str
        verdict: str
        confidence: int
        reason: str

    class MockValidatorOutput(BM):
        query: str
        overall_reliability: str
        confidence_score: int
        fact_checks: list[MockFactCheck]
        key_insights: list[str]
        gaps_identified: list[str]
        recommendation: str

    mock_retriever = [
        MockSearchResult(
            keyword="Electric vehicle trends India 2025",
            summary="India's EV market soared to 2.3 million units in 2025, representing 8% of total vehicle sales.",
            key_facts=[
                "India's EV market soared to 2.3 million units in 2025",
                "8% of total vehicle sales in 2025 were electric vehicles",
                "Tata, Mahindra, MG, and Kia are launching new electric models",
                "Growth driven by policy shifts and evolving consumer awareness"
            ]
        ),
        MockSearchResult(
            keyword="Government initiatives for electric vehicles in India",
            summary="The Indian government launched PM E-DRIVE scheme and FAME-II subsidies to promote EV adoption.",
            key_facts=[
                "PM E-DRIVE scheme launched in 2024",
                "FAME-II subsidies available for electric vehicle buyers",
                "Import duty rationalizations and PLI incentives provided",
                "Government aims to make India a global EV leader"
            ]
        )
    ]

    mock_validator = MockValidatorOutput(
        query="What are the latest trends in electric vehicles in India in 2025?",
        overall_reliability="high",
        confidence_score=80,
        fact_checks=[
            MockFactCheck(fact="India's EV market soared to 2.3 million units in 2025", verdict="confirmed", confidence=90, reason="Multiple sources confirm"),
            MockFactCheck(fact="8% of total vehicle sales in 2025 were electric vehicles", verdict="confirmed", confidence=85, reason="Source 1 confirms"),
            MockFactCheck(fact="PM E-DRIVE scheme launched in 2024", verdict="uncertain", confidence=60, reason="Date conflict between sources"),
            MockFactCheck(fact="FAME-II subsidies available for EV buyers", verdict="confirmed", confidence=90, reason="Source 3 confirms"),
        ],
        key_insights=[
            "India's EV market is growing rapidly with 2.3 million units in 2025",
            "Policy support is a key driver of EV adoption in India",
            "Multiple manufacturers are actively entering the Indian EV market"
        ],
        gaps_identified=[
            "Limited data on EV market forecast beyond 2025",
            "No specific pricing data for new EV models"
        ],
        recommendation="Research is highly reliable with minor date discrepancies to verify."
    )

    test_query = "What are the latest trends in electric vehicles in India in 2025?"

    print("=" * 60)
    print("  WRITER AGENT — SwarmIQ")
    print("=" * 60)
    print(f"Query: {test_query}")
    print("Generating final report...\n")

    report = run_writer(test_query, mock_retriever, mock_validator)
    print_report(report)
    print("\nWriter Agent done. All 4 agents complete!")