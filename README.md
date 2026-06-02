# SwarmIQ — Multi-Agent Research Assistant

> Microsoft Build Hackathon 2026 — Theme 05: Agent Swarms

SwarmIQ is a multi-agent research system where four specialized AI agents 
collaborate to answer any research question with validated, sourced, 
structured reports.

## How It Works

A single user query triggers a swarm of 4 agents running in sequence:
User Query
│
▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PLANNER   │────▶│  RETRIEVER  │────▶│  VALIDATOR  │────▶│   WRITER    │
│             │     │             │     │             │     │             │
│ Breaks query│     │ Searches web│     │ Cross-checks│     │ Produces    │
│ into tasks  │     │ for each    │     │ every fact  │     │ structured  │
│ & keywords  │     │ keyword     │     │ scores conf │     │ report      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘

## Agents

| Agent | Role | Model |
|-------|------|-------|
| Planner | Decomposes query into sub-tasks and search keywords | LLaMA 3.1 8B |
| Retriever | Searches DuckDuckGo for each keyword, extracts facts | LLaMA 3.1 8B |
| Validator | Cross-checks facts, scores confidence 0-100 | LLaMA 3.1 8B |
| Writer | Produces sectioned report with executive summary | LLaMA 3.3 70B |

## Setup

### Requirements
- Python 3.11+
- Groq API key (free at console.groq.com)

### Install

```bash
git clone https://github.com/vinsharmavin89-alw/swarm-research-agent
cd swarm-research-agent
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### Configure

```bash
# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

### Run

```bash
uvicorn main:app --reload
```

Open `dashboard.html` in your browser.

### Docker

```bash
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check, agent list |
| `/research` | POST | Run full 4-agent pipeline |
| `/plan-only` | POST | Run Planner agent only |
| `/docs` | GET | Interactive API documentation |

### Example Request

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are EV trends in India 2025?"}'
```

## Project Structure

swarm-research-agent/
├── main.py              # FastAPI app, pipeline orchestrator
├── dashboard.html       # Frontend UI
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-service orchestration
├── requirements.txt     # Dependencies
├── .env                 # API keys (never committed)
└── agents/
├── planner.py       # Agent 1: Query decomposition
├── retriever.py     # Agent 2: Web search + extraction
├── validator.py     # Agent 3: Fact validation
└── writer.py        # Agent 4: Report generation

## Tech Stack

- **LLM**: Groq (LLaMA 3.1 8B, LLaMA 3.3 70B)
- **Search**: DuckDuckGo (ddgs)
- **API**: FastAPI + Uvicorn
- **Validation**: Pydantic v2
- **Container**: Docker + Docker Compose
- **Frontend**: Vanilla HTML/CSS/JS

## Team

| Name | Role |
|------|------|
| Vinit | Full Stack + AI Architecture |

## Judging Criteria Addressed

| Criteria | How SwarmIQ addresses it |
|----------|--------------------------|
| AI Integration (25pts) | 4 specialized LLM agents with distinct roles |
| System Architecture (25pts) | Orchestrated pipeline, Pydantic contracts, Docker |
| Presentation + UX (15pts) | Real-time dashboard, agent status visualization |
| Prototype Readiness (15pts) | Live deployed URL, full working demo |
| Problem Depth (10pts) | Fact validation layer — unique differentiator |
| Market Fit (10pts) | Replaces hours of manual research |

