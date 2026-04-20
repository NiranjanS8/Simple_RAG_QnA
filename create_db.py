from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

loader = PyPDFLoader("document_loaders/Spring-Notes.pdf")

docs = loader.load() 

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 20
)

chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

print("Chroma DB created successfully.")
