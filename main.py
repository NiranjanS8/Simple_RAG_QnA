from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import uvicorn

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma-db"
TEMPLATES_DIR = BASE_DIR / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Minimal RAG PDF App")

embedding_model = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3"
)
llm = ChatMistralAI(model="mistral-small-2506")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for question-answering tasks.\n"
            "Use ONLY the provided context to answer the question.\n"
            "Do not use prior knowledge.\n"
            "If the answer is not in the context, say:\n"
            "\"I could not find the answer in the document.\"\n\n"
            "Be concise and accurate.\n"
            "If possible, quote relevant parts of the context."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        ),
    ]
)


def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
    )


def index_pdf(pdf_path: Path) -> int:
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    if not docs:
        raise ValueError("No readable content was found in the uploaded PDF.")

    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["uploaded_file"] = pdf_path.name

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    return len(chunks)


def answer_question(question: str) -> dict:
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 12, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "I could not find the answer in the document.",
            "sources": [],
        }

    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt_template.invoke({"context": context, "question": question})
    response = llm.invoke(final_prompt)

    sources = []
    for doc in docs:
        source_name = doc.metadata.get("uploaded_file") or doc.metadata.get("source") or "Uploaded PDF"
        page = doc.metadata.get("page")
        source_label = f"{source_name} (page {page + 1})" if isinstance(page, int) else source_name
        if source_label not in sources:
            sources.append(source_label)

    return {"answer": response.content, "sources": sources}


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files were uploaded.")

    indexed_files = []
    total_chunks = 0

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        unique_name = f"{uuid4().hex}_{Path(file.filename).name}"
        saved_path = UPLOAD_DIR / unique_name
        file_bytes = await file.read()
        saved_path.write_bytes(file_bytes)

        try:
            chunk_count = index_pdf(saved_path)
        except Exception as exc:
            saved_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}: {exc}") from exc

        indexed_files.append(file.filename)
        total_chunks += chunk_count

    file_list = ", ".join(indexed_files)
    return JSONResponse(
        {
            "message": f"Indexed {len(indexed_files)} PDF(s) into {total_chunks} chunks: {file_list}",
        }
    )


@app.post("/ask")
async def ask_question(payload: dict) -> JSONResponse:
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}") from exc

    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
