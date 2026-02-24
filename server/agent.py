"""Latin tutor agent using Microsoft Agent Framework + Azure Foundry.

Includes per-textbook search tools (FAISS + SQLite hybrid search),
a Whitaker's Words dictionary lookup tool, and a read_passage tool
that gives the agent access to the currently-displayed Latin text
with LatinCy grammatical annotations.
"""

import json
import logging
import os
import re
import sqlite3
import sys
import time
from collections.abc import Awaitable, Mapping
from pathlib import Path
from typing import Callable

import numpy as np
import faiss
import requests as http_requests
from agent_framework import FunctionInvocationContext, FunctionMiddleware, tool
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential, DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

from nlp import get_nlp

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1024"))
EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

TOP_K_PER_INDEX = 3

# Local database paths
DATA_DIR = Path(__file__).parent / os.getenv("LOCAL_DB_DIR", "data")
DB_PATH = DATA_DIR / "lector.db"
FAISS_PATH = DATA_DIR / "lector.faiss"

# ---------------------------------------------------------------------------
# Book descriptions — each entry maps a book key to its display name and a
# natural-language description surfaced to the agent via tool docstrings.
# ---------------------------------------------------------------------------

INDEXES = {
    "bradleys-arnold": {
        "display_name": "Bradley's Arnold",
        "description": (
            "An advanced Latin prose composition textbook. Covers complex and "
            "nuanced syntactic structures: conditional sentences, indirect speech, "
            "oratio obliqua, advanced relative clauses, and stylistic preferences "
            "of classical authors. Best for questions about sophisticated or "
            "advanced prose constructions, not basic grammar rules."
        ),
    },
    "lane-morgan-grammar": {
        "display_name": "Lane & Morgan Latin Grammar",
        "description": (
            "A formal reference grammar of the Latin language. Provides detailed "
            "descriptions of the uses of all cases, tenses, moods, and "
            "conjugations — the 'rules' of Latin grammar. Covers declensions, "
            "verb inflection, irregular forms, and phonology. Best for questions "
            "about specific grammatical rules and morphology, not complex syntax "
            "or prose style."
        ),
    },
    "north-prose-comp": {
        "display_name": "North & Hillard Prose Composition",
        "description": (
            "An introductory Latin prose composition textbook with model sentences "
            "and exercises. Covers basic style and sentence structure: purpose "
            "clauses, result clauses, temporal clauses, relative clauses, wishes, "
            "commands, and common constructions. Best for questions about "
            "fundamental prose composition and rendering English idioms into Latin."
        ),
    },
    "traupman-conversational": {
        "display_name": "Traupman Conversational Latin",
        "description": (
            "A phrasebook of neo-Latin neologisms and naturalistic colloquial "
            "conversation. Covers modern words and phrases rendered into Latin, "
            "everyday greetings, and contemporary topics (technology, sports, "
            "dining, travel). Best for converting modern concepts or colloquial "
            "ways of saying things into advisable classical Latin. Do NOT search "
            "this for formal grammar rules, declensions, or prose composition."
        ),
    },
}

# ---------------------------------------------------------------------------
# Embedding client (for search queries)
# ---------------------------------------------------------------------------

_embedding_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(
    _embedding_credential, "https://cognitiveservices.azure.com/.default"
)

_embedding_client = AzureOpenAI(
    azure_deployment=EMBEDDING_DEPLOYMENT,
    api_version=EMBEDDING_API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT,
    azure_ad_token_provider=_token_provider,
)

# ---------------------------------------------------------------------------
# Local database + FAISS index
# ---------------------------------------------------------------------------

_db_conn = None
_faiss_index = None
_search_available = False

if DB_PATH.exists() and FAISS_PATH.exists():
    try:
        _db_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _db_conn.row_factory = sqlite3.Row
        _faiss_index = faiss.read_index(str(FAISS_PATH))
        _search_available = True
        logger.info(
            "Textbook search loaded: %s (%d vectors)",
            DB_PATH, _faiss_index.ntotal,
        )
    except Exception as e:
        logger.warning("Failed to load textbook search index: %s", e)
else:
    logger.info(
        "Textbook search index not found at %s — search tools disabled. "
        "Run build_local_db.py to create the index.",
        DATA_DIR,
    )

# ---------------------------------------------------------------------------
# Per-book search tools
# ---------------------------------------------------------------------------


def _make_search_fn(book_key: str, display_name: str, description: str) -> Callable:
    """Create a search tool function for a specific textbook."""

    tool_name = f"search_{book_key.replace('-', '_')}"
    tool_description = (
        f"Search {display_name} for relevant passages.\n\n"
        f"{description}\n\n"
        f"Args:\n"
        f"    query: The search query describing what information to find.\n\n"
        f"Returns:\n"
        f"    Relevant passages with page citations from {display_name}."
    )

    @tool(name=tool_name, description=tool_description, approval_mode="never_require")
    def search_fn(query: str) -> str:
        t0 = time.perf_counter()
        embedding_response = _embedding_client.embeddings.create(
            input=[query],
            model=EMBEDDING_DEPLOYMENT,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        query_vector = np.array(
            embedding_response.data[0].embedding, dtype=np.float32
        ).reshape(1, -1)
        logger.debug("Search embedding: %.2fs", time.perf_counter() - t0)

        t1 = time.perf_counter()

        book_chunk_ids = [
            r[0] for r in _db_conn.execute(
                "SELECT id FROM chunks WHERE book_key = ?", (book_key,)
            ).fetchall()
        ]
        if not book_chunk_ids:
            return f"No passages indexed for {display_name}."

        # FAISS vector search (oversample, then filter to this book)
        k_search = TOP_K_PER_INDEX * 10
        distances, ids = _faiss_index.search(
            query_vector, min(k_search, _faiss_index.ntotal)
        )

        book_id_set = set(book_chunk_ids)
        vector_results = []
        for dist, chunk_id in zip(distances[0], ids[0]):
            if chunk_id == -1:
                continue
            if chunk_id in book_id_set:
                vector_results.append((chunk_id, float(dist)))
            if len(vector_results) >= TOP_K_PER_INDEX:
                break

        # FTS5 keyword search
        fts_results = _db_conn.execute(
            """
            SELECT c.id, rank
            FROM chunks_fts fts
            JOIN chunks c ON c.id = fts.rowid
            WHERE chunks_fts MATCH ? AND c.book_key = ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, book_key, TOP_K_PER_INDEX),
        ).fetchall()

        # Merge and deduplicate
        seen = set()
        merged_ids = []
        for chunk_id, _dist in vector_results:
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged_ids.append(chunk_id)
        for row in fts_results:
            if row[0] not in seen:
                seen.add(row[0])
                merged_ids.append(row[0])

        if not merged_ids:
            logger.debug(
                "Search %s: 0 results (%.2fs)", display_name, time.perf_counter() - t1
            )
            return f"No relevant passages found in {display_name}."

        placeholders = ",".join("?" for _ in merged_ids)
        rows = _db_conn.execute(
            f"SELECT id, page_number, content FROM chunks WHERE id IN ({placeholders})",
            merged_ids,
        ).fetchall()

        row_map = {r[0]: r for r in rows}
        passages = []
        for chunk_id in merged_ids:
            r = row_map.get(chunk_id)
            if r:
                passages.append(f"[{display_name}, p. {r[1]}]\n{r[2]}")

        logger.debug(
            "Search %s: %d results (%.2fs)",
            display_name, len(passages), time.perf_counter() - t1,
        )
        return "\n\n---\n\n".join(passages)

    return search_fn


# Build per-book search tools (only if the index is available)
search_tools = []
if _search_available:
    search_tools = [
        _make_search_fn(key, info["display_name"], info["description"])
        for key, info in INDEXES.items()
    ]

# ---------------------------------------------------------------------------
# Whitaker's Words dictionary lookup
# ---------------------------------------------------------------------------

WHITAKERS_API_BASE = "https://whitakers-words.com/api/translate/latin"


def _format_whitakers_entry(entry: dict) -> str:
    """Format a single Whitaker's Words API entry into readable text."""
    parts = []

    for rl in entry.get("rootLines", []):
        header = rl.get("root", "")
        pos = rl.get("partOfSpeech", "")
        version = rl.get("version", "")
        kind = rl.get("kind") or ""
        label = " ".join(filter(None, [pos, version, kind]))
        if label:
            header += f"  ({label})"
        parts.append(header)

    matches = entry.get("recordMatches", [])
    if matches:
        forms = []
        for m in matches:
            tokens = []
            for key in ("declension", "conjugation", "tense", "voice", "mood",
                        "case", "number", "gender", "person"):
                if m.get(key):
                    tokens.append(m[key])
            if tokens:
                forms.append(" ".join(tokens))
        if forms:
            parts.append("Forms: " + "; ".join(forms))

    meanings = [m.strip() for m in entry.get("meanings", []) if m.strip()]
    if meanings:
        parts.append("Meanings: " + ", ".join(meanings))

    return "\n".join(parts)


@tool(approval_mode="never_require")
def lookup_latin_word(word: str) -> str:
    """Look up a Latin word in Whitaker's Words dictionary.

    Returns dictionary entries with principal parts, part of speech,
    morphological forms, and English meanings. Use this tool when the
    student asks about the meaning, declension, conjugation, or parsing
    of a specific Latin word.

    Args:
        word: A single Latin word to look up.

    Returns:
        Dictionary entries from Whitaker's Words with forms and definitions.
    """
    t0 = time.perf_counter()
    url = f"{WHITAKERS_API_BASE}/{word.lower().strip()}"
    try:
        resp = http_requests.get(url, timeout=10)
        resp.raise_for_status()
    except http_requests.RequestException as exc:
        logger.warning("Whitaker's Words error: %s", exc)
        return f"Could not reach the Whitaker's Words dictionary: {exc}"

    data = resp.json()
    logger.debug("Whitaker's '%s': %d entries (%.2fs)", word, len(data), time.perf_counter() - t0)

    if not isinstance(data, list) or len(data) == 0:
        return f"No dictionary entries found for '{word}'."

    sections = [_format_whitakers_entry(e) for e in data]
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Text reading — manifest and section parsing
# ---------------------------------------------------------------------------

TEXTS_DIR = Path(__file__).parent.parent / "texts"
_MANIFEST_PATH = TEXTS_DIR / "manifest.json"

_text_manifest: dict[str, dict] = {}
if _MANIFEST_PATH.exists():
    with open(_MANIFEST_PATH, encoding="utf-8") as f:
        for entry in json.load(f):
            _text_manifest[entry["id"]] = entry
    logger.info("Text manifest loaded: %d texts", len(_text_manifest))

# Section marker pattern: a number at the start of the text or after whitespace,
# followed by a space and a capital letter (matches the UI's section rendering).
_SECTION_RE = re.compile(r"(?:^|\s)(\d+)\s+(?=[A-ZÀ-Ý])")


def _split_into_sections(text: str) -> list[tuple[int, str]]:
    """Split text into (section_number, section_text) pairs."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [(0, text.strip())]

    sections = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((num, section_text))
    return sections


def _format_morph(morph_str: str) -> str:
    """Convert spaCy morph string like 'Case=Abl|Number=Sing' to 'Abl.Sing'."""
    if not morph_str:
        return ""
    parts = []
    for pair in morph_str.split("|"):
        kv = pair.split("=")
        if len(kv) == 2:
            parts.append(kv[1])
    return ".".join(parts)


@tool(approval_mode="never_require")
def read_passage(text_id: str, sections: str = "all") -> str:
    """Read sections of the Latin text the student is currently viewing, with
    word-by-word grammatical analysis from LatinCy.

    Use this tool whenever the student asks about a specific word, line,
    section, or passage in the text they are reading. The context header
    in each message provides the text_id.

    Args:
        text_id: The text identifier (e.g., "boethius-3-p-4"), provided in
                 the [Text ID: ...] header of the user's message.
        sections: Which sections to read. Use "all" for the full text, a
                  single number like "11" for one section, or a range like
                  "11-14" for multiple sections.

    Returns:
        The requested text sections with per-word grammatical annotations
        (lemma, part of speech, morphological features).
    """
    if text_id not in _text_manifest:
        available = ", ".join(_text_manifest.keys()) if _text_manifest else "(none loaded)"
        return f"Unknown text ID '{text_id}'. Available texts: {available}"

    entry = _text_manifest[text_id]
    filepath = TEXTS_DIR / entry["file"]
    if not filepath.exists():
        return f"Text file not found: {entry['file']}"

    raw_text = filepath.read_text(encoding="utf-8")
    all_sections = _split_into_sections(raw_text)

    # Parse the requested section range
    if sections.strip().lower() == "all":
        selected = all_sections
    else:
        try:
            if "-" in sections:
                start_s, end_s = sections.split("-", 1)
                start_num, end_num = int(start_s.strip()), int(end_s.strip())
            else:
                start_num = end_num = int(sections.strip())
            selected = [(n, t) for n, t in all_sections if start_num <= n <= end_num]
        except ValueError:
            selected = all_sections

    if not selected:
        section_nums = [str(n) for n, _ in all_sections]
        return (
            f"No sections matched '{sections}'. "
            f"Available sections: {', '.join(section_nums)}"
        )

    # Cap at 20 sections to keep output manageable
    if len(selected) > 20:
        selected = selected[:20]

    # Build the text output
    output_parts = [f'=== {entry["title"]} ===\n']
    combined_text = ""
    for num, text in selected:
        label = f"§{num}" if num > 0 else ""
        output_parts.append(f"{label} {text}\n")
        combined_text += text + " "

    # Run LatinCy analysis
    output_parts.append("\n=== Word Analysis ===")
    try:
        nlp = get_nlp()
        doc = nlp(combined_text.strip())
        for token in doc:
            if token.pos_ == "PUNCT" or not token.text.strip():
                continue
            morph = _format_morph(str(token.morph))
            lemma_part = f" ({token.lemma_})" if token.lemma_ != token.text.lower() else ""
            morph_part = f" {morph}" if morph else ""
            output_parts.append(f"  {token.text} → {token.lemma_} [{token.pos_}]{morph_part}")
    except Exception as e:
        logger.warning("LatinCy analysis failed: %s", e)
        output_parts.append("  (LatinCy analysis unavailable)")

    return "\n".join(output_parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_textbook_list = "\n".join(
    f"- **{info['display_name']}**: {info['description']}"
    for info in INDEXES.values()
)

SYSTEM_PROMPT = f"""\
You are a friendly and knowledgeable Latin tutor. You help students read and \
understand classical Latin texts.

You have access to these tools:
- **read_passage**: Read the Latin text the student is currently viewing, with \
word-by-word grammatical analysis (lemma, part of speech, morphological features) \
from LatinCy, a pre-trained spaCy model. Note: these automated tags are sometimes \
wrong — use your own Latin knowledge and the grammar textbooks to verify when \
something looks off. If the student disagrees with a tagging, take their reading \
seriously. \
Each message includes a [Text ID: ...] header — pass this to read_passage. You can \
request specific sections (e.g., sections="11" or sections="11-14") or the full text \
(sections="all").
- **Textbook search tools**: Search four authoritative Latin textbooks for relevant \
passages on grammar, prose composition, and usage.
- **lookup_latin_word**: Look up a word in Whitaker's Words dictionary for principal \
parts, morphological analysis, and definitions.

Available textbooks:
{_textbook_list}

When answering questions, do not rely on your pretrained knowledge alone — always \
consult your tools to ground your answers in authoritative sources. Do not wait \
for the student to ask you to look something up; proactively use read_passage, \
textbook search, and dictionary lookup as needed.

1. When the student asks about a specific word, line, or section in the text, \
ALWAYS use read_passage first to see the actual text and its grammatical analysis. \
Do not guess or rely on truncated excerpts.
2. Use the textbook search tools when the question involves grammar rules, prose \
composition, or Latin usage — choose the textbook(s) most likely to help.
3. Use lookup_latin_word for vocabulary questions about specific words.
4. **Only cite sources you have actually consulted via a tool call.** Never \
reference a textbook or dictionary entry unless you searched for it or looked it \
up in this conversation. When you do cite, name the textbook and page number.
5. **Be concise.** Give direct, focused answers — a few sentences is usually \
enough. Do not repeat the student's question back to them. Avoid lengthy \
introductions, exhaustive lists, or tangential information. If the student wants \
more detail, they will ask.
"""


# ---------------------------------------------------------------------------
# Tool call notification middleware
# ---------------------------------------------------------------------------

TOOL_LABELS = {
    "read_passage": ("📖", "Reading passage"),
    "search_bradleys_arnold": ("🔍", "Searching Bradley's Arnold"),
    "search_lane_morgan_grammar": ("🔍", "Searching Lane & Morgan Grammar"),
    "search_north_prose_comp": ("🔍", "Searching North & Hillard"),
    "search_traupman_conversational": ("🔍", "Searching Traupman"),
    "lookup_latin_word": ("📚", "Looking up word"),
}


class ToolCallNotifier(FunctionMiddleware):
    """Middleware that records tool-call events during a single agent run."""

    def __init__(self):
        self._calls: list[dict] = []

    def clear(self):
        """Reset the call list before a new agent run."""
        self._calls.clear()

    def get_calls(self) -> list[dict]:
        """Return the collected tool-call records."""
        return list(self._calls)

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        name = context.function.name
        icon, base_label = TOOL_LABELS.get(name, ("🔧", name))

        # Include the query/word arg in the label when available
        args = context.arguments
        if isinstance(args, Mapping):
            args_dict = dict(args)
        elif hasattr(args, "model_dump"):
            args_dict = args.model_dump()
        else:
            args_dict = {}

        label = base_label
        if name == "lookup_latin_word" and args_dict.get("word"):
            label = f'Looking up "{args_dict["word"]}"'
        elif name == "read_passage" and args_dict.get("sections", "all") != "all":
            label = f'Reading §{args_dict["sections"]}'

        self._calls.append({"tool_call": name, "icon": icon, "label": label})
        await call_next()


# Singleton notifier shared between agent.py and app.py
tool_notifier = ToolCallNotifier()


def create_agent():
    """Create and return the Latin tutor agent with search tools."""
    credential = AzureCliCredential()
    client = AzureOpenAIResponsesClient(
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment_name=os.environ.get(
            "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME", "gpt-4.1"
        ),
        credential=credential,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )

    tools = [read_passage, *search_tools, lookup_latin_word]

    return client.as_agent(
        name="LatinTutor",
        instructions=SYSTEM_PROMPT,
        tools=tools,
        middleware=[tool_notifier],
    )
