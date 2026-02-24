# Lector

**Lector** (_Latin: "reader"_) is an interactive Latin reading environment. It renders classical Latin texts with clickable words, dictionary lookups, NLP-powered grammatical analysis, and an AI tutor chat — all in the browser.

## Features

- **Text Library** — Browse a growing collection of Latin texts via a dropdown selector.
- **Dictionary Lookup** — Click any word to see its dictionary entry (powered by [Whitaker's Words](https://whitakers-words.com/)).
- **Context Analysis** — Automatic morphological tagging (lemma, part of speech, case, number, etc.) via [LatinCy](https://huggingface.co/latincy), a spaCy-based Latin NLP pipeline. Grammatical features are displayed as colored tags in the sidebar, categorized by type (case, number, gender, tense, mood, voice, etc.).
- **AI Latin Tutor** — Chat panel (bottom of the UI) where you can ask questions about the text or any selected word. The tutor is context-aware: it knows which text you're reading and which word you have selected. It is instructed to be concise and to always cite the sources it consults.
- **Tool-Call Transparency** — When the AI tutor uses a tool (searching a textbook, reading the passage, looking up a word), a colored indicator pill appears in the chat feed so the user always knows when a source is being consulted.
- **Textbook Search (RAG)** — The AI tutor can search four authoritative Latin textbooks using hybrid vector + keyword search (FAISS + SQLite FTS5) to ground its answers in reference material:
  - **Bradley's Arnold** — Advanced Latin prose composition (conditional sentences, indirect speech, oratio obliqua, stylistic preferences).
  - **Lane & Morgan Latin Grammar** — Formal reference grammar (cases, tenses, moods, conjugations, declensions, irregular forms).
  - **North & Hillard Prose Composition** — Introductory prose composition (purpose clauses, result clauses, temporal clauses, commands).
  - **Traupman Conversational Latin** — Neo-Latin phrasebook (modern vocabulary, colloquial expressions, everyday greetings).
- **Passage Reading Tool** — The tutor can read sections of the currently displayed text on demand, with word-by-word LatinCy annotations, so it never has to guess about what the student is looking at.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Browser (Vanilla JS)                                │
│  ┌────────┐ ┌────────────┐ ┌──────────────────────┐  │
│  │ Reader │ │  Sidebar   │ │     Chat Panel       │  │
│  │ Panel  │ │ (analysis) │ │ (AI tutor + tools)   │  │
│  └────────┘ └────────────┘ └──────────────────────┘  │
└──────────────────────┬───────────────────────────────┘
                       │  HTTP (JSON)
┌──────────────────────▼───────────────────────────────┐
│  FastAPI Server (app.py)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ POST /api/   │  │ POST /api/   │  │ GET /api/ │  │
│  │   chat       │  │   analyze    │  │   health  │  │
│  └──────┬───────┘  └──────────────┘  └───────────┘  │
│         │                                            │
│  ┌──────▼──────────────────────────────────────────┐ │
│  │  Agent (agent.py) — Microsoft Agent Framework   │ │
│  │  ┌────────────┐ ┌──────────┐ ┌───────────────┐  │ │
│  │  │ 4 Search   │ │ Lookup   │ │ Read Passage  │  │ │
│  │  │ Tools      │ │ Latin    │ │ (LatinCy NLP) │  │ │
│  │  │ (FAISS+DB) │ │ Word     │ │               │  │ │
│  │  └────────────┘ └──────────┘ └───────────────┘  │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

The chat endpoint (`POST /api/chat`) is **non-streaming**: the server runs the agent to completion, collects all tool calls and the final reply, and returns a single JSON response:

```json
{
  "session_id": "uuid",
  "tool_calls": [
    { "tool_call": "search_lane_morgan_grammar", "icon": "🔍", "label": "Searching Lane & Morgan Grammar" },
    { "tool_call": "read_passage", "icon": "📖", "label": "Reading §11" }
  ],
  "reply": "The word *multiplici* is an adjective in the ablative case…"
}
```

The frontend displays a "…" thinking indicator while waiting, then renders the tool-call indicators followed by the reply. This approach is simpler and more reliable than streaming, with the trade-off that the user waits for the full response before seeing any text.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JavaScript (ES modules), HTML, CSS |
| Backend | Python 3.10+, FastAPI, Uvicorn |
| NLP | spaCy + LatinCy (`la_core_web_lg`) |
| AI Tutor | Azure OpenAI (GPT-4.1) via the Microsoft Agent Framework |
| Textbook Search | FAISS (vector) + SQLite FTS5 (keyword) hybrid search |
| Embeddings | Azure OpenAI `text-embedding-3-small` (1024 dimensions) |
| Dictionary | Whitaker's Words REST API |

## Prerequisites

- **Python 3.10+**
- **Azure CLI** — authenticated (`az login`) with access to an Azure OpenAI resource
- **Azure OpenAI endpoint** with:
  - A deployed chat model (e.g., `gpt-4.1`) for the AI tutor
  - A `text-embedding-3-small` deployment for textbook search

## Setup

### 1. Install Python dependencies

```bash
cd server
pip install -r requirements.txt
```

### 2. Download the LatinCy spaCy model

```bash
pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-any-py3-none-any.whl
```

This is a ~200 MB model. It is loaded once on startup (via a singleton in `nlp.py`) and shared between the `/api/analyze` endpoint and the `read_passage` tool.

### 3. Configure environment variables

Copy the example file and fill in your Azure OpenAI details:

```bash
cp server/.env.example server/.env
```

Edit `server/.env`:

```env
# Required
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME=gpt-4.1

# Required for textbook search
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_DIMENSIONS=1024

# Optional — only needed if re-indexing textbooks that require OCR
# AZURE_AI_ENDPOINT=https://your-ai-services-resource.services.ai.azure.com
```

Authentication uses `AzureCliCredential` (for the chat model) and `DefaultAzureCredential` (for embeddings), so make sure you're logged in with `az login`.

### 4. Start the server

```bash
cd server
uvicorn app:app --reload --port 8000
```

The app is now available at **http://localhost:8000**.

On first startup, you'll see log messages confirming which tools loaded:
- `Textbook search loaded: ... (N vectors)` — search tools are active.
- `Textbook search index not found` — the tutor works but cannot search textbooks. The chat UI will show a warning.

### 5. Textbook search index (optional)

The AI tutor's textbook search requires a pre-built index in `server/data/`. If the index files (`lector.db` and `lector.faiss`) are present, search tools are enabled automatically.

To rebuild the index (e.g., after adding a new textbook to `server/textbooks/`):

```bash
cd server
python build_local_db.py                    # index all textbooks
python build_local_db.py bradleys-arnold    # re-index a single textbook
```

This requires the embedding env vars in `server/.env` and an `az login` session. The indexer:
1. Extracts text from source files in `server/textbooks/` (PDF via PyMuPDF, or plain `.txt`; OCR via Azure Content Understanding when needed)
2. Chunks the text into passages (~500 tokens each)
3. Generates embeddings via Azure OpenAI
4. Stores chunks + metadata in `lector.db` (SQLite with FTS5) and vectors in `lector.faiss`

## Usage

1. **Select a text** from the dropdown in the header.
2. **Click any Latin word** to see its dictionary definition and grammatical analysis in the sidebar. Morphological features appear as colored tags (e.g., blue for case, green for number, purple for gender).
3. **Ask the AI tutor** questions in the chat panel at the bottom. The tutor knows which text you're reading and which word you have selected.

The tutor will:
- **Read the passage** when you ask about a specific word or line
- **Search textbooks** when you ask about grammar rules or constructions
- **Look up words** in the dictionary for vocabulary questions
- **Cite its sources** — you'll see which textbook (with page numbers), dictionary, or passage analysis it consulted

When you switch texts, the chat session resets so the tutor's context stays accurate.

## Managing the Text Library

Texts live in the `texts/` directory. To add a new text:

1. Place your `.txt` file in `texts/` (e.g., `texts/cicero-cat-1.txt`).
2. Add an entry to `texts/manifest.json`:

```json
{
  "id": "cicero-cat-1",
  "title": "Cicero — In Catilinam I",
  "file": "cicero-cat-1.txt"
}
```

The manifest is **not** auto-generated — you maintain it manually so you control display titles and ordering. The dropdown in the UI is populated directly from this file.

Texts can include section markers (a number followed by a capital letter, e.g., `11 Atque ut...`). The `read_passage` tool uses these markers to let the tutor read specific sections on demand.

## Project Structure

```
├── index.html              # Main HTML page
├── css/
│   └── style.css           # All styles (morph tags, tool indicators, chat, etc.)
├── js/
│   ├── main.js             # App entry point, text loading, dropdown
│   ├── reader.js           # Tokenizer, word click handling, sidebar rendering
│   ├── api.js              # Whitaker's Words + LatinCy API calls
│   ├── chat.js             # Chat panel, sends context (textId, selectedWord)
│   ├── markdown.js         # Lightweight markdown → HTML for chat bubbles
│   └── store.js            # localStorage cache for dictionary lookups
├── texts/
│   ├── manifest.json       # Text library metadata (id, title, file)
│   └── *.txt               # Latin text files (currently Boethius III.1–4)
└── server/
    ├── app.py              # FastAPI server: /api/chat, /api/analyze, /api/health
    ├── agent.py            # Agent definition, tools, system prompt, middleware
    ├── nlp.py              # Shared LatinCy spaCy model singleton
    ├── build_local_db.py   # SQLite + FAISS indexing pipeline
    ├── index_textbooks.py  # Text extraction and chunking from source files
    ├── content_understanding_client.py  # Azure AI Content Understanding (OCR)
    ├── test_agent.py       # Smoke test for the agent
    ├── requirements.txt    # Python dependencies
    ├── .env.example        # Environment variable template
    ├── .env                # Your local config (git-ignored)
    ├── data/               # Search index: lector.db + lector.faiss (git-ignored)
    └── textbooks/          # Source textbook files (git-ignored)
```

### Key server modules

- **`app.py`** — FastAPI application. Three endpoints: `/api/chat` (non-streaming JSON, runs agent to completion), `/api/analyze` (LatinCy morphological analysis), `/api/health` (reports search availability). Serves the frontend as static files.
- **`agent.py`** — Creates the AI tutor agent using the Microsoft Agent Framework. Defines six tools (`read_passage`, four per-textbook search functions, `lookup_latin_word`), the system prompt, and a `ToolCallNotifier` middleware that records which tools were called during each interaction.
- **`nlp.py`** — Loads the `la_core_web_lg` LatinCy model once and shares it between the `/api/analyze` endpoint and the `read_passage` tool.
