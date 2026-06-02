from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import traceback

from agents.planner import run_planner
from agents.retriever import run_retriever
from agents.validator import run_validator
from agents.writer import run_writer, print_report

app = FastAPI(
    title="SwarmIQ",
    description="Multi-Agent Research Assistant — Microsoft Build Hackathon",
    version="1.0.0"
)

# Allow frontend to call this API later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class PipelineStatus(BaseModel):
    step: str
    status: str
    message: str


@app.get("/")
def root():
    return {
        "name": "SwarmIQ",
        "description": "Multi-Agent Research Assistant",
        "agents": {
            "1_planner": "Breaks query into research plan",
            "2_retriever": "Searches web for each keyword",
            "3_validator": "Validates and scores each fact",
            "4_writer": "Produces final structured report"
        },
        "status": "ready",
        "version": "1.0.0"
    }


@app.post("/research")
def run_full_pipeline(request: QueryRequest):
    """
    The main endpoint — runs all 4 agents in sequence.

    Flow:
      User query
        → Planner (creates research plan)
        → Retriever (searches web for each keyword)
        → Validator (checks facts, scores confidence)
        → Writer (produces final report)
        → Returns complete result
    """

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(query) < 10:
        raise HTTPException(status_code=400, detail="Query too short — please be more specific")

    pipeline_log = []
    start_time = datetime.now()

    try:
        # ── STEP 1: PLANNER ──────────────────────────────
        print(f"\n[SwarmIQ] Query received: {query}")
        print("[SwarmIQ] Step 1/4 — Planner Agent running...")

        plan = run_planner(query)
        pipeline_log.append({
            "step": "planner",
            "status": "success",
            "sub_tasks": len(plan.sub_tasks),
            "keywords": len(plan.search_keywords),
            "complexity": plan.complexity
        })
        print(f"[SwarmIQ] Planner done — {len(plan.search_keywords)} keywords, complexity: {plan.complexity}")

        # ── STEP 2: RETRIEVER ─────────────────────────────
        print("[SwarmIQ] Step 2/4 — Retriever Agent running...")

        retriever_output = run_retriever(query, plan.search_keywords)
        total_facts = sum(len(r.key_facts) for r in retriever_output.results)
        pipeline_log.append({
            "step": "retriever",
            "status": "success",
            "keywords_searched": retriever_output.total_keywords_searched,
            "total_facts_found": total_facts
        })
        print(f"[SwarmIQ] Retriever done — {total_facts} facts found across {retriever_output.total_keywords_searched} searches")

        # ── STEP 3: VALIDATOR ─────────────────────────────
        print("[SwarmIQ] Step 3/4 — Validator Agent running...")

        validation = run_validator(query, retriever_output.results)
        confirmed = sum(1 for fc in validation.fact_checks if fc.verdict == "confirmed")
        pipeline_log.append({
            "step": "validator",
            "status": "success",
            "facts_checked": len(validation.fact_checks),
            "confirmed": confirmed,
            "overall_reliability": validation.overall_reliability,
            "confidence_score": validation.confidence_score
        })
        print(f"[SwarmIQ] Validator done — {confirmed}/{len(validation.fact_checks)} facts confirmed, reliability: {validation.overall_reliability}")

        # ── STEP 4: WRITER ────────────────────────────────
        print("[SwarmIQ] Step 4/4 — Writer Agent running...")

        report = run_writer(query, retriever_output.results, validation)
        pipeline_log.append({
            "step": "writer",
            "status": "success",
            "sections": len(report.sections),
            "title": report.title
        })
        print(f"[SwarmIQ] Writer done — report: '{report.title}'")

        # ── BUILD FINAL RESPONSE ──────────────────────────
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[SwarmIQ] Pipeline complete in {elapsed:.1f}s\n")

        return {
            "success": True,
            "query": query,
            "elapsed_seconds": round(elapsed, 1),
            "pipeline_log": pipeline_log,

            # Planner output
            "plan": {
                "sub_tasks": plan.sub_tasks,
                "search_keywords": plan.search_keywords,
                "complexity": plan.complexity
            },

            # Retriever output
            "research": [
                {
                    "keyword": r.keyword,
                    "summary": r.summary,
                    "key_facts": r.key_facts
                }
                for r in retriever_output.results
            ],

            # Validator output
            "validation": {
                "overall_reliability": validation.overall_reliability,
                "confidence_score": validation.confidence_score,
                "recommendation": validation.recommendation,
                "fact_checks": [
                    {
                        "fact": fc.fact,
                        "verdict": fc.verdict,
                        "confidence": fc.confidence,
                        "reason": fc.reason
                    }
                    for fc in validation.fact_checks
                ],
                "key_insights": validation.key_insights,
                "gaps_identified": validation.gaps_identified
            },

            # Final report
            "report": {
                "title": report.title,
                "executive_summary": report.executive_summary,
                "sections": report.sections,
                "confidence_score": report.confidence_score,
                "reliability": report.reliability,
                "gaps": report.gaps,
                "generated_at": report.generated_at
            }
        }

    except Exception as e:
        print(f"[SwarmIQ] Pipeline error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "pipeline_log": pipeline_log,
                "failed_at": pipeline_log[-1]["step"] if pipeline_log else "planner"
            }
        )


@app.post("/plan-only")
def plan_only(request: QueryRequest):
    """Test just the Planner Agent."""
    result = run_planner(request.query)
    return result.model_dump()