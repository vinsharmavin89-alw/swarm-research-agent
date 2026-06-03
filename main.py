from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
import traceback

from agents.planner import run_planner
from agents.retriever import run_retriever
from agents.validator import run_validator
from agents.writer import run_writer

app = FastAPI(
    title="SwarmIQ",
    description="Multi-Agent Research Assistant - Microsoft Build Hackathon",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


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


@app.get("/dashboard")
def dashboard():
    return FileResponse("dashboard.html")


@app.post("/research")
def run_full_pipeline(request: QueryRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(query) < 10:
        raise HTTPException(status_code=400, detail="Query too short - please be more specific")

    pipeline_log = []
    start_time = datetime.now()

    try:
        print(f"\n[SwarmIQ] Query: {query}")
        print("[SwarmIQ] Step 1/4 - Planner...")
        plan = run_planner(query)
        pipeline_log.append({
            "step": "planner",
            "status": "success",
            "sub_tasks": len(plan.sub_tasks),
            "keywords": len(plan.search_keywords),
            "complexity": plan.complexity
        })

        print("[SwarmIQ] Step 2/4 - Retriever...")
        retriever_output = run_retriever(query, plan.search_keywords)
        total_facts = sum(len(r.key_facts) for r in retriever_output.results)
        pipeline_log.append({
            "step": "retriever",
            "status": "success",
            "keywords_searched": retriever_output.total_keywords_searched,
            "total_facts_found": total_facts
        })

        print("[SwarmIQ] Step 3/4 - Validator...")
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

        print("[SwarmIQ] Step 4/4 - Writer...")
        report = run_writer(query, retriever_output.results, validation)
        pipeline_log.append({
            "step": "writer",
            "status": "success",
            "sections": len(report.sections),
            "title": report.title
        })

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[SwarmIQ] Done in {elapsed:.1f}s")

        return {
            "success": True,
            "query": query,
            "elapsed_seconds": round(elapsed, 1),
            "pipeline_log": pipeline_log,
            "plan": {
                "sub_tasks": plan.sub_tasks,
                "search_keywords": plan.search_keywords,
                "complexity": plan.complexity
            },
            "research": [
                {
                    "keyword": r.keyword,
                    "summary": r.summary,
                    "key_facts": r.key_facts
                }
                for r in retriever_output.results
            ],
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
        print(f"[SwarmIQ] Error: {str(e)}")
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
    result = run_planner(request.query)
    return result.model_dump()