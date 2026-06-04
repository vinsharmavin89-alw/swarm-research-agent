from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import traceback
import asyncio
import json
import uuid

from agents.planner import run_planner
from agents.retriever import run_retriever
from agents.validator import run_validator
from agents.writer import run_writer

app = FastAPI(
    title="SwarmIQ",
    description="Multi-Agent Research Assistant - Microsoft Build Hackathon 2026",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store for agent message logs
sessions = {}


class QueryRequest(BaseModel):
    query: str


def make_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event message."""
    payload = json.dumps({"type": event_type, "data": data, "ts": datetime.now().strftime("%H:%M:%S")})
    return f"data: {payload}\n\n"


@app.get("/")
def root():
    return {
        "name": "SwarmIQ v2.0",
        "description": "Multi-Agent Research Assistant",
        "agents": {
            "1_planner": "Breaks query into research plan",
            "2_retriever": "Parallel web search for each keyword",
            "3_validator": "Cross-checks facts with retry loop",
            "4_writer": "Produces final structured report"
        },
        "status": "ready",
        "version": "2.0.0",
        "features": ["streaming", "parallel_retrieval", "validator_retry", "agent_message_log"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "agents": ["planner", "retriever", "validator", "writer"],
        "uptime": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/dashboard")
def dashboard():
    return FileResponse("dashboard.html")


@app.get("/research/stream")
async def research_stream(query: str):
    """
    Streaming endpoint using Server-Sent Events.
    Frontend receives live updates as each agent runs.
    This is what makes the real-time dashboard possible.
    """

    async def event_generator():
        session_id = str(uuid.uuid4())[:8]
        messages = []

        def log(agent_from, agent_to, message, msg_type="info"):
            entry = {
                "from": agent_from,
                "to": agent_to,
                "message": message,
                "type": msg_type,
                "session": session_id
            }
            messages.append(entry)
            return entry

        try:
            yield make_event("session_start", {
                "session_id": session_id,
                "query": query,
                "message": "SwarmIQ swarm initializing..."
            })

            await asyncio.sleep(0.1)

            # ── PLANNER ──────────────────────────────────────────────
            yield make_event("agent_start", {
                "agent": "planner",
                "message": "Analyzing query and building research plan..."
            })

            log("system", "planner", f"Received query: '{query}'")
            yield make_event("agent_message", log("system", "planner", f"Query received: '{query}'"))
            await asyncio.sleep(0.1)

            plan = run_planner(query)

            yield make_event("agent_message", log("planner", "retriever",
                f"Research plan ready. Complexity: {plan.complexity}. Dispatching {len(plan.search_keywords)} keywords.", "success"))

            for i, kw in enumerate(plan.search_keywords):
                yield make_event("agent_message", log("planner", "retriever",
                    f"Keyword {i+1}: '{kw}'", "keyword"))
                await asyncio.sleep(0.05)

            yield make_event("agent_done", {
                "agent": "planner",
                "sub_tasks": plan.sub_tasks,
                "keywords": plan.search_keywords,
                "complexity": plan.complexity
            })

            await asyncio.sleep(0.1)

            # ── RETRIEVER (parallel) ──────────────────────────────────
            yield make_event("agent_start", {
                "agent": "retriever",
                "message": f"Launching {len(plan.search_keywords)} parallel web searches..."
            })

            yield make_event("agent_message", log("retriever", "web",
                f"Dispatching {len(plan.search_keywords)} searches in parallel", "parallel"))

            # Run retrieval (the actual search happens inside run_retriever)
            retriever_output = await asyncio.get_event_loop().run_in_executor(
                None, run_retriever, query, plan.search_keywords
            )

            total_facts = sum(len(r.key_facts) for r in retriever_output.results)

            for r in retriever_output.results:
                yield make_event("agent_message", log("web", "retriever",
                    f"'{r.keyword}' → {len(r.key_facts)} facts extracted", "data"))
                await asyncio.sleep(0.08)

            yield make_event("agent_message", log("retriever", "validator",
                f"Search complete. Passing {total_facts} facts for validation.", "success"))

            yield make_event("agent_done", {
                "agent": "retriever",
                "keywords_searched": retriever_output.total_keywords_searched,
                "total_facts": total_facts,
                "results": [{"keyword": r.keyword, "facts": r.key_facts} for r in retriever_output.results]
            })

            await asyncio.sleep(0.1)

            # ── VALIDATOR (with retry loop) ───────────────────────────
            yield make_event("agent_start", {
                "agent": "validator",
                "message": f"Cross-checking {total_facts} facts for accuracy..."
            })

            yield make_event("agent_message", log("validator", "facts",
                f"Beginning cross-validation of {total_facts} facts across {len(retriever_output.results)} sources", "check"))

            validation = await asyncio.get_event_loop().run_in_executor(
                None, run_validator, query, retriever_output.results
            )

            confirmed = sum(1 for fc in validation.fact_checks if fc.verdict == "confirmed")
            uncertain = sum(1 for fc in validation.fact_checks if fc.verdict == "uncertain")
            contradicted = sum(1 for fc in validation.fact_checks if fc.verdict == "contradicted")

            # Show individual fact verdicts
            for fc in validation.fact_checks[:5]:
                icon = "confirmed" if fc.verdict == "confirmed" else "contradicted" if fc.verdict == "contradicted" else "uncertain"
                yield make_event("agent_message", log("validator", "facts",
                    f"[{fc.verdict.upper()}] {fc.fact[:60]}... ({fc.confidence}%)", icon))
                await asyncio.sleep(0.06)

            # Retry loop if confidence is too low
            if validation.confidence_score < 50:
                yield make_event("agent_message", log("validator", "retriever",
                    f"Confidence {validation.confidence_score}% is below threshold. Requesting additional research.", "retry"))
                yield make_event("retry_triggered", {
                    "reason": f"Confidence score {validation.confidence_score}% < 50% threshold",
                    "action": "Retriever searching for supplementary data..."
                })
                # Get retry keywords from gaps
                retry_keywords = [g[:50] for g in validation.gaps_identified[:2]]
                if retry_keywords:
                    retry_output = await asyncio.get_event_loop().run_in_executor(
                        None, run_retriever, query, retry_keywords
                    )
                    # Merge results
                    retriever_output.results.extend(retry_output.results)
                    yield make_event("agent_message", log("retriever", "validator",
                        f"Supplementary search complete. Added {sum(len(r.key_facts) for r in retry_output.results)} more facts.", "success"))
                    # Re-validate
                    validation = await asyncio.get_event_loop().run_in_executor(
                        None, run_validator, query, retriever_output.results
                    )
                    yield make_event("agent_message", log("validator", "writer",
                        f"Re-validation complete. New confidence: {validation.confidence_score}%", "success"))

            yield make_event("agent_message", log("validator", "writer",
                f"Validation complete: {confirmed} confirmed, {uncertain} uncertain, {contradicted} contradicted. Confidence: {validation.confidence_score}%", "success"))

            yield make_event("agent_done", {
                "agent": "validator",
                "confidence_score": validation.confidence_score,
                "reliability": validation.overall_reliability,
                "confirmed": confirmed,
                "uncertain": uncertain,
                "contradicted": contradicted,
                "fact_checks": [{"fact": fc.fact, "verdict": fc.verdict, "confidence": fc.confidence, "reason": fc.reason} for fc in validation.fact_checks],
                "key_insights": validation.key_insights,
                "gaps_identified": validation.gaps_identified,
                "recommendation": validation.recommendation
            })

            await asyncio.sleep(0.1)

            # ── WRITER ────────────────────────────────────────────────
            yield make_event("agent_start", {
                "agent": "writer",
                "message": "Synthesizing validated research into final report..."
            })

            yield make_event("agent_message", log("writer", "output",
                f"Received {len(validation.fact_checks)} validated facts. Composing report sections.", "write"))

            report = await asyncio.get_event_loop().run_in_executor(
                None, run_writer, query, retriever_output.results, validation
            )

            yield make_event("agent_message", log("writer", "output",
                f"Report complete: '{report.title}' — {len(report.sections)} sections, {report.confidence_score}% confidence", "success"))

            yield make_event("agent_done", {
                "agent": "writer",
                "title": report.title,
                "sections": len(report.sections)
            })

            await asyncio.sleep(0.1)

            # ── FINAL RESULT ──────────────────────────────────────────
            yield make_event("pipeline_complete", {
                "query": query,
                "session_id": session_id,
                "agent_messages": messages,
                "plan": {
                    "sub_tasks": plan.sub_tasks,
                    "search_keywords": plan.search_keywords,
                    "complexity": plan.complexity
                },
                "research": [{"keyword": r.keyword, "summary": r.summary, "key_facts": r.key_facts} for r in retriever_output.results],
                "validation": {
                    "overall_reliability": validation.overall_reliability,
                    "confidence_score": validation.confidence_score,
                    "recommendation": validation.recommendation,
                    "fact_checks": [{"fact": fc.fact, "verdict": fc.verdict, "confidence": fc.confidence, "reason": fc.reason} for fc in validation.fact_checks],
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
            })

        except Exception as e:
            print(f"[SwarmIQ] Stream error: {str(e)}")
            traceback.print_exc()
            yield make_event("error", {
                "message": str(e),
                "detail": "Pipeline error — check server logs"
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/research")
def research_sync(request: QueryRequest):
    """Synchronous fallback endpoint."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        plan = run_planner(query)
        retriever_output = run_retriever(query, plan.search_keywords)
        validation = run_validator(query, retriever_output.results)
        report = run_writer(query, retriever_output.results, validation)

        return {
            "success": True,
            "query": query,
            "plan": {"sub_tasks": plan.sub_tasks, "search_keywords": plan.search_keywords, "complexity": plan.complexity},
            "research": [{"keyword": r.keyword, "summary": r.summary, "key_facts": r.key_facts} for r in retriever_output.results],
            "validation": {
                "overall_reliability": validation.overall_reliability,
                "confidence_score": validation.confidence_score,
                "recommendation": validation.recommendation,
                "fact_checks": [{"fact": fc.fact, "verdict": fc.verdict, "confidence": fc.confidence, "reason": fc.reason} for fc in validation.fact_checks],
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan-only")
def plan_only(request: QueryRequest):
    result = run_planner(request.query)
    return result.model_dump()