from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel
import json

load_dotenv()

client = Groq()


class PlannerOutput(BaseModel):
    original_query: str
    sub_tasks: list[str]
    search_keywords: list[str]
    complexity: str


def run_planner(user_query: str) -> PlannerOutput:
    """
    Planner Agent — breaks a user query into a research plan.
    Input:  raw user question (string)
    Output: PlannerOutput with sub_tasks and search_keywords
    """

    system_prompt = """You are a Planner Agent in a multi-agent research system.

Analyze the user's research query and produce a structured research plan.

You must respond with ONLY a raw JSON object. 
No markdown, no code fences, no explanation, no extra text whatsoever.
Just the JSON object starting with { and ending with }

The JSON must have exactly these four fields:

{
  "original_query": "the user's exact question as a string",
  "sub_tasks": ["specific research task 1", "specific research task 2", "specific research task 3"],
  "search_keywords": ["search phrase 1", "search phrase 2", "search phrase 3"],
  "complexity": "moderate"
}

Rules:
- sub_tasks: 2 to 4 items. Each is a specific thing that needs researching.
- search_keywords: 3 to 5 items. Each is an exact phrase someone would type into Google.
- complexity: must be exactly one of these three words: simple, moderate, complex
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research query: {user_query}"}
        ]
    )

    raw_json = response.choices[0].message.content

    # Clean up in case model adds markdown fences anyway
    raw_json = raw_json.strip()
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()
    import re
    raw_json = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw_json)
    data = json.loads(raw_json)
    return PlannerOutput(**data)


if __name__ == "__main__":
    test_query = "What are the latest trends in electric vehicles in India in 2025?"

    print("=" * 55)
    print("  PLANNER AGENT — SwarmIQ")
    print("=" * 55)
    print(f"Query: {test_query}\n")

    result = run_planner(test_query)

    print(f"Complexity : {result.complexity}")
    print(f"\nSub-tasks ({len(result.sub_tasks)}):")
    for i, task in enumerate(result.sub_tasks, 1):
        print(f"  {i}. {task}")
    print(f"\nSearch keywords ({len(result.search_keywords)}):")
    for kw in result.search_keywords:
        print(f"  - {kw}")
    print("\nPlanner Agent done. Ready to pass to Retriever.")