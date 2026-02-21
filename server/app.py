"""FastAPI server exposing the Latin tutor agent as a chat endpoint."""

import logging
import pathlib
import uuid

import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import create_agent

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
_nlp = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


def _get_nlp():
    """Lazy-load the LatinCy spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("la_core_web_lg")
        _nlp.max_length = 2_500_000
    return _nlp


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

    if ctx.get("text"):
        # Send a brief excerpt so the model has passage context
        text = ctx["text"]
        if len(text) > 1500:
            text = text[:1500] + "…"
        parts.append(f"[Passage context: {text}]")

    parts.append(req.message)
    return "\n".join(parts)


class AnalyzeRequest(BaseModel):
    text: str


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """Run LatinCy on the passage and return per-token annotations."""
    try:
        nlp = _get_nlp()
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
        agent = _get_agent()
    except Exception as e:
        logger.exception("Failed to create agent")
        return StreamingResponse(
            iter([f"[Error] Could not initialize the AI tutor: {e}\n"]),
            media_type="text/plain",
        )

    # Resolve or create session
    sid = req.session_id or str(uuid.uuid4())
    if sid not in _sessions:
        try:
            _sessions[sid] = agent.create_session()
        except Exception as e:
            logger.exception("Failed to create session")
            return StreamingResponse(
                iter([f"[Error] Could not create a session: {e}\n"]),
                media_type="text/plain",
            )
    session = _sessions[sid]

    user_msg = _build_user_message(req)

    async def stream_response():
        yield f'{{"session_id": "{sid}"}}\n'
        try:
            async for chunk in agent.run(user_msg, session=session, stream=True):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.exception("Error during agent streaming")
            yield f"\n[Error] The AI service encountered a problem: {e}"

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.get("/api/health")
async def health():
    """Quick check that the agent can be created and credentials are valid."""
    try:
        _get_agent()
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Health check failed")
        return {"status": "error", "detail": str(e)}


# Serve frontend static files (must be after API routes)
_frontend_dir = pathlib.Path(__file__).resolve().parent.parent
app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
