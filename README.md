# Simple RAG QnA

A simple Retrieval-Augmented Generation (RAG) PDF question-answering app built with FastAPI, LangChain, ChromaDB, and Mistral.

This project lets you:

- Upload one or more PDF files
- Convert PDF content into embeddings
- Store the embeddings in ChromaDB
- Ask questions based only on the uploaded documents

## Tech Stack

- Python
- FastAPI
- LangChain
- ChromaDB
- Mistral AI
- Hugging Face embeddings

## Project Structure

```text
Simple_RAG_QnA/
|-- main.py
|-- create_db.py
|-- requirements.txt
|-- .env
|-- templates/
|-- uploads/
|-- chroma-db/
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/NiranjanS8/Simple_RAG_QnA.git
cd Simple_RAG_QnA
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your API keys:

```env
MISTRAL_API_KEY=your_mistral_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
```

## Run the App

```bash
python main.py
```

The app will start at:

```text
http://127.0.0.1:8000
```

## How It Works

1. Upload PDF files
2. Split the content into chunks
3. Generate embeddings for each chunk
4. Store the chunks in ChromaDB
5. Retrieve relevant chunks for a question
6. Generate an answer using the retrieved context

## Notes

- Answers are generated only from the uploaded document context
- Uploaded files and local vector data are not meant to be committed to GitHub
- `.env`, `uploads/`, and `chroma-db/` are ignored in `.gitignore`
