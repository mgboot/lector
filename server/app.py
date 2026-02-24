"""FastAPI server exposing the Latin tutor agent as a chat endpoint."""

import asyncio
import logging
import pathlib
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import create_agent, tool_notifier, _search_available
from nlp import get_nlp

logger = logging.getLogger(__name__)

app = FastAPI(title="Lector Latin Tutor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id → AgentSession
_sessions: dict[str, object] = {}
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    context: dict | None = None  # { text, selectedWord, wordData }


class ChatResponse(BaseModel):
    reply: str
    session_id: str


def _build_user_message(req: ChatRequest) -> str:
    """Prepend context info to the user's message so the agent has it."""
    parts = []
    ctx = req.context or {}

    if ctx.get("textTitle"):
        parts.append(f"[Currently reading: {ctx['textTitle']}]")

    if ctx.get("textId"):
        parts.append(f"[Text ID: {ctx['textId']}]")

    if ctx.get("selectedWord"):
        parts.append(f"[Selected word: {ctx['selectedWord']}]")

    if ctx.get("wordData"):
        wd = ctx["wordData"]
        if isinstance(wd, list):
            for entry in wd:
                meanings = entry.get("meanings", [])
                root_lines = entry.get("rootLines", [])
                if root_lines:
                    rl = root_lines[0]
                    parts.append(
                        f"[Dictionary: root=\"{rl.get('root', '')}\", "
                        f"pos={rl.get('partOfSpeech', '')}, "
                        f"gender={rl.get('gender', '')}]"
                    )
                if meanings:
                    parts.append(f"[Meanings: {'; '.join(m.strip() for m in meanings if m.strip())}]")
                matches = entry.get("recordMatches", [])
                if matches:
                    forms = []
                    for rm in matches:
                        f = " ".join(
                            filter(None, [rm.get("case"), rm.get("number"), rm.get("gender")])
                        )
                        if f:
                            forms.append(f)
                    if forms:
                        parts.append(f"[Forms: {', '.join(forms)}]")

    parts.append(req.message)
    return "\n".join(parts)


class AnalyzeRequest(BaseModel):
    text: str


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Run LatinCy on the passage and return per-token annotations."""
    try:
        nlp = get_nlp()
    except Exception as e:
        logger.exception("Failed to load LatinCy model")
        return {"error": f"Could not load LatinCy model: {e}"}

    doc = nlp(req.text)
    tokens = [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "morph": str(token.morph),
            "start": token.idx,
            "end": token.idx + len(token.text),
        }
        for token in doc
    ]
    return {"tokens": tokens}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        agent = await asyncio.to_thread(_get_agent)
    except Exception as e:
        logger.exception("Failed to create agent")
        return {"reply": f"[Error] Could not initialize the AI tutor: {e}", "session_id": "", "tool_calls": []}

    # Resolve or create session
    sid = req.session_id or str(uuid.uuid4())
    if sid not in _sessions:
        try:
            _sessions[sid] = agent.create_session()
        except Exception as e:
            logger.exception("Failed to create session")
            return {"reply": f"[Error] Could not create a session: {e}", "session_id": sid, "tool_calls": []}
    session = _sessions[sid]

    user_msg = _build_user_message(req)

    # Clear any stale tool-call records
    tool_notifier.clear()

    try:
        result = await agent.run(user_msg, session=session)
        reply = result.text if hasattr(result, 'text') and result.text else ""
    except Exception as e:
        logger.exception("Error during agent run")
        reply = f"[Error] The AI service encountered a problem: {e}"

    calls = tool_notifier.get_calls()

    return {
        "session_id": sid,
        "tool_calls": calls,
        "reply": reply.strip() if reply else "The tutor didn't respond. Check that the server is running and configured correctly.",
    }


@app.get("/api/health")
def health():
    """Quick check that the agent can be created and credentials are valid."""
    try:
        _get_agent()
        return {
            "status": "ok",
            "search_available": _search_available,
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {"status": "error", "detail": str(e)}


# Serve frontend static files (must be after API routes)
_frontend_dir = pathlib.Path(__file__).resolve().parent.parent
app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
