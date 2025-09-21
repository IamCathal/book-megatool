

import logging
from fastapi import FastAPI, UploadFile, File, Header, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger("epub-semantic-search")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

DEFAULT_CHUNK_SIZE = 1000


def printDetailsForBook(book):
    if book is None:
        print("book is null")
        return

    title = book.get_metadata('DC', 'title')[0][0]
    author = book.get_metadata('DC', 'creator')[0][0]
    isbn = book.get_metadata('DC', 'identifier')[0][0]

    logger.info(f"Book details: Title: '{title}' Author: '{author}' ISBN: '{isbn}'")


def extractContentFromEpub(filePath: Path):
    logger.info(f"Extracting content from epub: {filePath}")

    book = epub.read_epub(filePath)
    printDetailsForBook(book)

    text = []
    for document in book.get_items():
        if document.get_type() == ebooklib.ITEM_DOCUMENT:
            textContent = clean_html(document.get_content().decode('utf-8'))
            text.append(textContent)

    logger.info(f"Extracted {len(text)} document sections from epub.")
    return ' '.join(text)


def splitIntoSections(text: str, chunk_size: int):
    logger.info("Splitting text into sections...")
    paragraphs = text.split("\n\n")
    sections = []
    currentSection = ''
    for thisParagraph in paragraphs:
        if len(thisParagraph) > 80:
            currentSection += ' ' + thisParagraph.strip()
            if len(currentSection) >= chunk_size:
                sections.append(currentSection)
                currentSection = ''
    if currentSection:
        sections.append(currentSection)
    logger.info(f"Split into {len(sections)} sections.")
    return sections


def clean_html(html_content: str):
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup(['script', 'style']):
        element.decompose()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n\n'.join(chunk for chunk in chunks if chunk)

    return text


def find_similar_sections(query, sections, top_n=5):
    logger.info(f"Finding similar sections for query: '{query}' (top_n={top_n})")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    logger.info("Encoding query...")
    query_vector = model.encode([query])[0]

    logger.info(f"Encoding {len(sections)} sections...")
    section_vectors = model.encode(sections)

    logger.info("Calculating similarities...")
    similarities = np.dot(section_vectors, query_vector) / (
        np.linalg.norm(section_vectors, axis=1) * np.linalg.norm(query_vector)
    )

    top_indices = np.argsort(similarities)[-top_n:]
    logger.info(f"Top indices: {top_indices[::-1]}")

    return [(float(similarities[i]), sections[i]) for i in top_indices[::-1]]


@app.post("/search-epub/")
async def search_epub(
    req: Request,
    file: UploadFile = File(...),
):
    if not file.filename.endswith(".epub"):
        logger.warning("File upload rejected: not an .epub file")
        raise HTTPException(status_code=400, detail="File must be .epub")

    query = req.headers.get("X-QUERY")
    top_n = int(req.headers.get("X-TOP-N"))
    chunk_size = int(req.headers.get("X-CHUNK-SIZE"))

    logger.info(f"Received file upload: {file.filename}, query: '{query}', top_n: {top_n}, chunk_size: {chunk_size}")

    if chunk_size < 40:
        chunk_size = DEFAULT_CHUNK_SIZE
    logger.info(f"Received chunk size {chunk_size} < 40, using default value of {DEFAULT_CHUNK_SIZE} instead")

    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
        file_bytes = await file.read()
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    logger.info(f"Saved uploaded file to temp path: {tmp_path}")
    try:
        bookContent = extractContentFromEpub(tmp_path)
        logger.info(f"Book content length: {len(bookContent)} characters")
        sections = splitIntoSections(bookContent, chunk_size)
        results = find_similar_sections(query, sections, top_n=top_n)
        # Only return preview of section (first 1000 chars)
        response = [
            {"similarity": sim, "section_preview": sec[:1000]} for sim, sec in results
        ]
        logger.info(f"Returning {len(response)} results.")
    
        return JSONResponse(content={"results": response})

    except Exception as e:
        logger.exception(f"Error processing epub: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        tmp_path.unlink(missing_ok=True)
        logger.info(f"Deleted temp file: {tmp_path}")


@app.get("/healthz")
def health_check():
    logger.info("Health check requested.")
    return {"status": "ok"}
