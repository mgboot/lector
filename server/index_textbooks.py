"""
index_textbooks.py

Extracts text from four Latin textbooks (3 PDFs + 1 .txt), splits into
paragraph/section-based chunks, generates embeddings via Azure OpenAI,
and indexes into four separate Azure AI Search indexes.

Authentication: API key-based for both Azure Search and Azure OpenAI.
"""

import os
import re
import json
import hashlib
import tempfile
from pathlib import Path

import fitz  # pymupdf
import requests
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from content_understanding_client import ContentUnderstandingClient

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient, SearchIndexingBufferedSender
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SimpleField,
        SearchFieldDataType,
        SearchableField,
        SearchField,
        VectorSearch,
        HnswAlgorithmConfiguration,
        VectorSearchProfile,
        SemanticConfiguration,
        SemanticPrioritizedFields,
        SemanticField,
        SemanticSearch,
        SearchIndex,
        AzureOpenAIVectorizer,
        AzureOpenAIVectorizerParameters,
    )
    _HAS_AZURE_SEARCH = True
except ImportError:
    _HAS_AZURE_SEARCH = False

from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(override=True)

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", "")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY", "")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1024"))
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

# Map each textbook file to its index name
TEXTBOOKS = {
    "Bradley's Arnold.pdf": "bradleys-arnold",
    "lane-morgan-latin-grammar.txt": "lane-morgan-grammar",
    "Latin Prose Composition - North.pdf": "north-prose-comp",
    "Traupman Conversational-Latin-for-Oral-Proficiency.pdf": "traupman-conversational",
}

TEXTBOOKS_DIR = Path(__file__).parent / "textbooks"

# Azure AI Content Understanding (OCR fallback for scanned PDFs)
CONTENT_UNDERSTANDING_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", OPENAI_ENDPOINT)
CONTENT_UNDERSTANDING_API_VERSION = "2025-11-01"

# If a PDF yields fewer than this many total characters, treat it as scanned
SCANNED_PDF_CHAR_THRESHOLD = 1000

# Chunking parameters
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 2000
EMBEDDING_BATCH_SIZE = 16  # number of chunks per embedding API call

# ---------------------------------------------------------------------------
# Auth — API key for Search, Entra ID for OpenAI
# ---------------------------------------------------------------------------

search_credential = AzureKeyCredential(SEARCH_ADMIN_KEY) if (_HAS_AZURE_SEARCH and SEARCH_ADMIN_KEY) else None

openai_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(openai_credential, "https://cognitiveservices.azure.com/.default")

openai_client = AzureOpenAI(
    azure_deployment=EMBEDDING_DEPLOYMENT,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT,
    azure_ad_token_provider=token_provider,
) if OPENAI_ENDPOINT else None

index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=search_credential) if search_credential else None

_cu_client = None


def _get_cu_client():
    """Lazy-init the Content Understanding client (only needed for OCR)."""
    global _cu_client
    if _cu_client is None:
        _cu_client = ContentUnderstandingClient(
            endpoint=CONTENT_UNDERSTANDING_ENDPOINT or OPENAI_ENDPOINT,
            api_version=CONTENT_UNDERSTANDING_API_VERSION,
            token_provider=token_provider,
        )
        _cu_client.ensure_defaults({
            "gpt-4.1": "gpt-4.1",
            "gpt-4.1-mini": "gpt-4.1-mini",
            "text-embedding-3-large": "text-embedding-3-large",
        })
    return _cu_client

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_pages_from_pdf(pdf_path: Path) -> list[dict]:
    """Return a list of {'page': int, 'text': str} for each page."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": i, "text": text})
    doc.close()
    return pages


CU_MAX_PAGES = 300  # Content Understanding page limit per request


def _ocr_single_pdf(pdf_path: str) -> list[dict]:
    """OCR a single PDF (must be ≤ CU_MAX_PAGES) and return page dicts."""
    cu_client = _get_cu_client()
    response = cu_client.begin_analyze_binary(
        analyzer_id="prebuilt-documentSearch",
        file_path=pdf_path,
    )
    result = cu_client.poll_result(response)

    contents = result.get("result", {}).get("contents", [])
    if not contents:
        return []

    markdown = contents[0].get("markdown", "")
    if not markdown.strip():
        return []

    page_texts = re.split(r"<!--\s*PageBreak\s*-->", markdown)
    pages = []
    for i, text in enumerate(page_texts, start=1):
        text = text.strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def extract_pages_with_ocr(pdf_path: Path) -> list[dict]:
    """Use Azure AI Content Understanding to OCR a scanned PDF.

    PDFs exceeding CU_MAX_PAGES are split into batches automatically.
    """
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if total_pages <= CU_MAX_PAGES:
        doc.close()
        return _ocr_single_pdf(str(pdf_path))

    # Split into batches of CU_MAX_PAGES and OCR each
    all_pages: list[dict] = []
    for start in range(0, total_pages, CU_MAX_PAGES):
        end = min(start + CU_MAX_PAGES, total_pages)
        print(f"    OCR batch: pages {start + 1}–{end} of {total_pages}")
        batch_doc = fitz.open()
        batch_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(tmp_fd)
        batch_doc.save(tmp_path)
        batch_doc.close()
        try:
            batch_pages = _ocr_single_pdf(tmp_path)
            # Re-number pages to reflect their position in the full PDF
            for p in batch_pages:
                p["page"] += start
            all_pages.extend(batch_pages)
        finally:
            os.unlink(tmp_path)

    doc.close()
    return all_pages


def extract_pages_from_txt(txt_path: Path) -> list[dict]:
    """Treat the whole file as page 1 and split into synthetic pages of ~4000 chars."""
    with open(txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    # Split into ~4000-char blocks on line boundaries for initial page-like grouping
    lines = full_text.splitlines(keepends=True)
    pages = []
    current = []
    current_len = 0
    page_num = 1
    for line in lines:
        current.append(line)
        current_len += len(line)
        if current_len >= 4000:
            pages.append({"page": page_num, "text": "".join(current)})
            current = []
            current_len = 0
            page_num += 1
    if current:
        pages.append({"page": page_num, "text": "".join(current)})
    return pages


# ---------------------------------------------------------------------------
# Chunking — paragraph/section-based
# ---------------------------------------------------------------------------

# Splitting heuristic: double newlines, or lines that look like section headers
_SPLIT_PATTERN = re.compile(r"\n\s*\n")


def chunk_text(text: str) -> list[str]:
    """Split text into paragraph-based chunks, merging short ones and splitting long ones."""
    raw_paragraphs = _SPLIT_PATTERN.split(text)
    chunks: list[str] = []
    buffer = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = (buffer + "\n\n" + para).strip() if buffer else para

        if len(candidate) > MAX_CHUNK_CHARS:
            # Flush buffer first if it has content
            if buffer:
                chunks.append(buffer)
                buffer = ""
            # Split the oversized paragraph on sentence boundaries
            for sub in _split_long(para):
                chunks.append(sub)
        elif len(candidate) < MIN_CHUNK_CHARS:
            buffer = candidate
        else:
            buffer = candidate

    if buffer:
        chunks.append(buffer)

    return chunks


def _split_long(text: str) -> list[str]:
    """Split a long paragraph into sub-chunks on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) > MAX_CHUNK_CHARS and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def pages_to_chunks(pages: list[dict]) -> list[dict]:
    """Convert page-level data into search-ready chunks with metadata."""
    all_chunks = []
    idx = 0
    for page_info in pages:
        for chunk_text_str in chunk_text(page_info["text"]):
            all_chunks.append({
                "chunk_index": idx,
                "page_number": page_info["page"],
                "content": chunk_text_str,
            })
            idx += 1
    return all_chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """Add 'contentVector' to each chunk dict using Azure OpenAI."""
    texts = [c["content"] for c in chunks]
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

    for i, chunk in enumerate(chunks):
        chunk["contentVector"] = all_embeddings[i]

    return chunks


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------


def create_search_index(index_name: str) -> None:
    """Create or update the search index with vector + semantic config."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, sortable=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, sortable=True, filterable=True),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnswProfile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[
            VectorSearchProfile(
                name="hnswProfile",
                algorithm_configuration_name="hnsw",
                vectorizer_name="openaiVectorizer",
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="openaiVectorizer",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=OPENAI_ENDPOINT,
                    deployment_name=EMBEDDING_DEPLOYMENT,
                    model_name=EMBEDDING_DEPLOYMENT,
                ),
            )
        ],
    )

    semantic_config = SemanticConfiguration(
        name="semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")]
        ),
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )
    result = index_client.create_or_update_index(index)
    print(f"  Index '{result.name}' created/updated")


# ---------------------------------------------------------------------------
# Document upload
# ---------------------------------------------------------------------------


def upload_documents(index_name: str, chunks: list[dict]) -> None:
    """Upload chunked documents to the search index."""
    # Prepare documents with a stable id
    documents = []
    for chunk in chunks:
        doc_id = hashlib.md5(
            f"{index_name}-{chunk['chunk_index']}".encode()
        ).hexdigest()
        documents.append({
            "id": doc_id,
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "page_number": chunk["page_number"],
            "contentVector": chunk["contentVector"],
        })

    with SearchIndexingBufferedSender(
        endpoint=SEARCH_ENDPOINT,
        index_name=index_name,
        credential=search_credential,
    ) as batch_client:
        batch_client.upload_documents(documents=documents)

    print(f"  Uploaded {len(documents)} documents to '{index_name}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_textbook(filename: str, index_name: str) -> None:
    """End-to-end pipeline for a single textbook."""
    filepath = TEXTBOOKS_DIR / filename
    print(f"\n{'='*60}")
    print(f"Processing: {filename} -> index '{index_name}'")
    print(f"{'='*60}")

    # 1. Extract text
    print("  Extracting text...")
    if filepath.suffix.lower() == ".pdf":
        pages = extract_pages_from_pdf(filepath)
        total_chars = sum(len(p["text"]) for p in pages)
        if total_chars < SCANNED_PDF_CHAR_THRESHOLD:
            print(f"  Only {total_chars} chars extracted — PDF appears to be scanned, using Content Understanding OCR...")
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
    chunks = generate_embeddings(chunks)

    # 4. Create index
    print("  Creating search index...")
    create_search_index(index_name)

    # 5. Upload documents
    print("  Uploading documents...")
    upload_documents(index_name, chunks)

    print(f"  Done: '{index_name}' is ready for search!")


def main():
    import argparse

    if not _HAS_AZURE_SEARCH or not SEARCH_ADMIN_KEY:
        print("ERROR: Azure AI Search credentials not configured.")
        print("Set AZURE_SEARCH_SERVICE_ENDPOINT and AZURE_SEARCH_ADMIN_KEY in .env")
        print("(This script indexes into Azure AI Search. For local indexing, use build_local_db.py)")
        return

    parser = argparse.ArgumentParser(description="Index Latin textbooks into Azure AI Search")
    parser.add_argument(
        "index", nargs="?", default=None,
        help="Index name to (re)process (e.g. north-prose-comp). Omit to process all.",
    )
    args = parser.parse_args()

    print("Lector Textbooks — Azure AI Search Indexing")
    print("Using API key for Search, Entra ID for OpenAI\n")
    print(f"Search endpoint: {SEARCH_ENDPOINT}")
    print(f"OpenAI endpoint: {OPENAI_ENDPOINT}")
    print(f"Embedding model: {EMBEDDING_DEPLOYMENT} ({EMBEDDING_DIMENSIONS}d)")

    if args.index:
        # Find the matching textbook by index name
        match = {f: idx for f, idx in TEXTBOOKS.items() if idx == args.index}
        if not match:
            valid = ", ".join(TEXTBOOKS.values())
            parser.error(f"Unknown index '{args.index}'. Valid indexes: {valid}")
        for filename, index_name in match.items():
            process_textbook(filename, index_name)
    else:
        for filename, index_name in TEXTBOOKS.items():
            process_textbook(filename, index_name)

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
