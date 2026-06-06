## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Inference | Groq API (LLaMA 3.1 8B, LLaMA 3.3 70B) — Azure AI Foundry compatible |
| AI Development | GitHub Copilot (AI-assisted development) |
| CI/CD Pipeline | GitHub Actions (Microsoft infrastructure) |
| API Framework | FastAPI + Uvicorn |
| Agent Contracts | Pydantic v2 |
| Web Search | DuckDuckGo DDGS |
| Container | Docker + docker-compose |
| Deployment | Render Cloud |
| Frontend | HTML/CSS/JS |

## Microsoft Stack Integration

SwarmIQ is built with Microsoft developer tools at every layer:

- **GitHub Copilot** — Used throughout development for AI-assisted code generation
- **GitHub Actions** — Automated CI/CD pipeline tests all 4 agents on every push
- **GitHub** — Version control, public repository, issue tracking
- **Azure AI Compatible** — Agent inference layer uses OpenAI-compatible API format, enabling one-line switch to Azure AI Foundry endpoint

To switch to Azure AI Foundry, change one line in `.env`:
```
AZURE_OPENAI_ENDPOINT=your-azure-endpoint
AZURE_OPENAI_KEY=your-azure-key
```