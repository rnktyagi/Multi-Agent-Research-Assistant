from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from src.graph import workflow


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's research question.")


class ResearchResponse(BaseModel):
    report: str
    objectives: list[str]
    is_approved: bool
    improvements: list[str]
    retries_used: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Research assistant graph compiled and ready.")
    app.state.research_graph = workflow.compile()
    yield
    print("Shutting down.")


app = FastAPI(title="Multi-Agent Research Assistant", lifespan=lifespan)


@app.post("/research", response_model=ResearchResponse)
async def run_research(payload: ResearchRequest, request: Request):
    research_graph = request.app.state.research_graph  # <-- pull it out of app.state

    try:
        result = await research_graph.ainvoke({
            "user_query": payload.query,
            "retry": 0,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research pipeline failed: {e}")

    return ResearchResponse(
        report=result.get("written_report", ""),
        objectives=result.get("objectives", []),
        is_approved=result.get("is_approved", False),
        improvements=result.get("improvements", []),
        retries_used=result.get("retry", 0),
    )