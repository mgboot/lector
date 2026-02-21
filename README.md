# Lector

**Lector** (_Latin: "reader"_) is an interactive Latin reading environment. It renders classical Latin texts with clickable words, dictionary lookups, NLP-powered grammatical analysis, and an AI tutor chat — all in the browser.

## Features

- **Text Library** — Browse a growing collection of Latin texts via a dropdown selector.
- **Dictionary Lookup** — Click any word to see its dictionary entry (powered by [Whitaker's Words](https://whitakers-words.com/)).
- **Context Analysis** — Automatic morphological tagging (lemma, part of speech, case, number, etc.) via [LatinCy](https://huggingface.co/latincy), a spaCy-based Latin NLP pipeline.
- **AI Latin Tutor** — Chat panel where you can ask questions about the text or any selected word. The tutor sees the same passage and word context you see, so its answers are grounded in what you're reading.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JavaScript (ES modules), HTML, CSS |
| Backend | Python 3.10+, FastAPI, Uvicorn |
| NLP | spaCy + LatinCy (`la_core_web_lg`) |
| AI Tutor | Azure OpenAI (GPT-4.1) via the Microsoft Agent Framework |
| Dictionary | Whitaker's Words REST API |

## Prerequisites

- **Python 3.10+**
- **Azure CLI** — authenticated (`az login`) with access to an Azure OpenAI resource
- **Azure OpenAI endpoint** with a deployed GPT model (e.g., `gpt-4.1`)

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

### 3. Configure environment variables

Copy the example file and fill in your Azure OpenAI details:

```bash
cp server/.env.example server/.env
```

Edit `server/.env`:

```
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME=gpt-4.1
```

Authentication uses `AzureCliCredential`, so make sure you're logged in with `az login`.

### 4. Start the server

```bash
cd server
uvicorn app:app --reload --port 8000
```

The app is now available at **http://localhost:8000**.

## Usage

1. **Select a text** from the dropdown in the header.
2. **Click any Latin word** to see its dictionary definition and grammatical analysis in the sidebar.
3. **Ask the AI tutor** questions in the chat panel at the bottom. The tutor is aware of which text you're reading and which word you have selected.

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

## Project Structure

```
├── index.html              # Main HTML page
├── css/
│   └── style.css           # Styles
├── js/
│   ├── main.js             # App entry point, text loading, dropdown
│   ├── reader.js           # Tokenizer, word click handling, sidebar
│   ├── api.js              # Whitaker's Words + LatinCy API calls
│   ├── chat.js             # Chat panel, context management
│   ├── markdown.js         # Markdown renderer for chat bubbles
│   └── store.js            # localStorage cache for word data
├── texts/
│   ├── manifest.json       # Text library metadata
│   ├── boethius-3-p-1.txt  # Boethius, Consolatio III, Prosa 1
│   └── boethius-3-p-2.txt  # Boethius, Consolatio III, Prosa 2
└── server/
    ├── app.py              # FastAPI server (chat, NLP, static files)
    ├── agent.py            # Azure OpenAI agent configuration
    ├── test_agent.py       # Smoke test for the agent
    ├── requirements.txt    # Python dependencies
    ├── .env.example        # Environment variable template
    └── .env                # Your local config (not committed)
```
