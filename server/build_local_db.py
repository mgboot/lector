"""
build_local_db.py

Extracts text from four Latin textbooks (3 PDFs + 1 .txt), splits into
paragraph/section-based chunks, generates embeddings via Azure OpenAI
(text-embedding-3-small), and stores everything in a local SQLite database
with a companion FAISS vector index.

Authentication: Entra ID for Azure OpenAI.

Usage:
    python build_local_db.py                    # process all textbooks
    python build_local_db.py bradleys-arnold    # process a single textbook
"""

import os
import sqlite3
import argparse
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

# Reuse text extraction and chunking from the existing indexing pipeline
from index_textbooks import (
    extract_pages_from_pdf,
    extract_pages_from_txt,
    extract_pages_with_ocr,
    pages_to_chunks,
    SCANNED_PDF_CHAR_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(override=True)

OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1024"))
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

TEXTBOOKS = {
    "Bradley's Arnold.pdf": {
        "key": "bradleys-arnold",
        "name": "Bradley's Arnold",
    },
    "lane-morgan-latin-grammar.txt": {
        "key": "lane-morgan-grammar",
        "name": "Lane & Morgan Latin Grammar",
    },
    "Latin Prose Composition - North.pdf": {
        "key": "north-prose-comp",
        "name": "North & Hillard Prose Composition",
    },
    "Traupman Conversational-Latin-for-Oral-Proficiency.pdf": {
        "key": "traupman-conversational",
        "name": "Traupman Conversational Latin",
    },
}

TEXTBOOKS_DIR = Path(__file__).parent / "textbooks"
DATA_DIR = Path(__file__).parent / os.getenv("LOCAL_DB_DIR", "data")
DB_PATH = DATA_DIR / "lector.db"
FAISS_PATH = DATA_DIR / "lector.faiss"

EMBEDDING_BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Auth — Entra ID for OpenAI
# ---------------------------------------------------------------------------

openai_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    openai_credential, "https://cognitiveservices.azure.com/.default"
)

openai_client = AzureOpenAI(
    azure_deployment=EMBEDDING_DEPLOYMENT,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT,
    azure_ad_token_provider=token_provider,
)

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create the SQLite database and tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_key TEXT NOT NULL,
            book_name TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_key);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            content='chunks',
            content_rowid='id'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
        END;
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using Azure OpenAI."""
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = openai_client.embeddings.create(
            input=batch,
            model=EMBEDDING_DEPLOYMENT,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        all_embeddings.extend([item.embedding for item in response.data])
        print(f"  Embedded {min(start + EMBEDDING_BATCH_SIZE, len(texts))}/{len(texts)} chunks")
    return all_embeddings


# ---------------------------------------------------------------------------
# FAISS index management
# ---------------------------------------------------------------------------


def save_faiss_index(index, faiss_path: Path):
    """Save the FAISS index to disk."""
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))


def rebuild_faiss_from_db(conn: sqlite3.Connection, faiss_path: Path, dimensions: int):
    """Rebuild the entire FAISS index from embeddings stored in SQLite."""
    rows = conn.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimensions))
    if rows:
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        vectors = np.array(
            [np.frombuffer(r[1], dtype=np.float32) for r in rows], dtype=np.float32
        )
        index.add_with_ids(vectors, ids)
    save_faiss_index(index, faiss_path)
    return index


# ---------------------------------------------------------------------------
# Textbook processing
# ---------------------------------------------------------------------------


def process_textbook(
    filename: str,
    book_key: str,
    book_name: str,
    conn: sqlite3.Connection,
    faiss_index,
) -> None:
    """End-to-end pipeline for a single textbook."""
    filepath = TEXTBOOKS_DIR / filename
    print(f"\n{'='*60}")
    print(f"Processing: {filename} -> '{book_key}'")
    print(f"{'='*60}")

    # 1. Extract text
    print("  Extracting text...")
    if filepath.suffix.lower() == ".pdf":
        pages = extract_pages_from_pdf(filepath)
        total_chars = sum(len(p["text"]) for p in pages)
        if total_chars < SCANNED_PDF_CHAR_THRESHOLD:
            print(f"  Only {total_chars} chars — PDF appears scanned, using OCR...")
            pages = extract_pages_with_ocr(filepath)
    else:
        pages = extract_pages_from_txt(filepath)
    print(f"  Extracted {len(pages)} pages")

    # 2. Chunk
    print("  Chunking text...")
    chunks = pages_to_chunks(pages)
    print(f"  Created {len(chunks)} chunks")

    # 3. Generate embeddings
    print("  Generating embeddings...")
    texts = [c["content"] for c in chunks]
    embeddings = generate_embeddings(texts)

    # 4. Clear any existing data for this book
    conn.execute("DELETE FROM chunks WHERE book_key = ?", (book_key,))
    conn.commit()

    # 5. Insert chunks into SQLite (with embeddings stored as BLOBs)
    print("  Inserting into SQLite...")
    chunk_ids = []
    for chunk, embedding in zip(chunks, embeddings):
        emb_blob = np.array(embedding, dtype=np.float32).tobytes()
        cursor = conn.execute(
            "INSERT INTO chunks (book_key, book_name, chunk_index, page_number, content, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (book_key, book_name, chunk["chunk_index"], chunk["page_number"], chunk["content"], emb_blob),
        )
        chunk_ids.append(cursor.lastrowid)
    conn.commit()

    # 6. Add vectors to FAISS index
    print("  Adding vectors to FAISS index...")
    vectors = np.array(embeddings, dtype=np.float32)
    ids = np.array(chunk_ids, dtype=np.int64)
    faiss_index.add_with_ids(vectors, ids)

    print(f"  Done: {len(chunks)} chunks indexed for '{book_key}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build local SQLite + FAISS index for Latin textbooks"
    )
    parser.add_argument(
        "book", nargs="?", default=None,
        help="Book key to (re)index (e.g. north-prose-comp). Omit to process all.",
    )
    args = parser.parse_args()

    print("Lector Textbooks — Local SQLite + FAISS Indexing")
    print(f"OpenAI endpoint : {OPENAI_ENDPOINT}")
    print(f"Embedding model : {EMBEDDING_DEPLOYMENT} ({EMBEDDING_DIMENSIONS}d)")
    print(f"Database        : {DB_PATH}")
    print(f"FAISS index     : {FAISS_PATH}")

    conn = init_db(DB_PATH)

    # Temporary FAISS index for collecting new vectors during processing
    faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIMENSIONS))

    if args.book:
        match = {f: info for f, info in TEXTBOOKS.items() if info["key"] == args.book}
        if not match:
            valid = ", ".join(info["key"] for info in TEXTBOOKS.values())
            parser.error(f"Unknown book '{args.book}'. Valid keys: {valid}")
        for filename, info in match.items():
            process_textbook(filename, info["key"], info["name"], conn, faiss_index)
    else:
        for filename, info in TEXTBOOKS.items():
            process_textbook(filename, info["key"], info["name"], conn, faiss_index)

    # Rebuild FAISS from all embeddings stored in SQLite
    print("\nRebuilding FAISS index from all stored embeddings...")
    final_index = rebuild_faiss_from_db(conn, FAISS_PATH, EMBEDDING_DIMENSIONS)
    print(f"FAISS index saved ({final_index.ntotal} vectors)")

    conn.close()
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
